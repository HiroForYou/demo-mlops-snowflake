# %% [markdown]
# # Alertas de Observabilidad
#
# Este notebook consulta las tablas de observabilidad del modelo
# (OBS_PERFORMANCE, OBS_DATA_DRIFT, OBS_PRED_DRIFT) para identificar
# registros con alertas (warning o critical) y generar un reporte
# consolidado de elementos preocupantes para un modelo dado.

# %% [markdown]
# ## 1. Setup
#
# Configuración inicial: sesión de Snowpark e imports.

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F
session = get_active_session()

# %% [markdown]
# ### 1A. Constants
#
# Constantes del proyecto: base de datos, esquema, tablas de observabilidad,
# nombre del modelo y etiquetas de nivel de alerta.

# %%
# Account info
DATABASE = "BD_AA_DEV"
FEATURES_SCHEMA = "SC_FEATURES_BMX"

session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {FEATURES_SCHEMA}").collect()

# Observability tables (all in SC_FEATURES_BMX)
OBS_PERFORMANCE_TABLE = "OBS_PERFORMANCE"
OBS_DATA_DRIFT_TABLE = "OBS_DATA_DRIFT"
OBS_PRED_DRIFT_TABLE = "OBS_PRED_DRIFT"

OBS_TABLES = {
    "performance": OBS_PERFORMANCE_TABLE,
    "data_drift_features": OBS_DATA_DRIFT_TABLE,
    "data_drift_segments": OBS_DATA_DRIFT_TABLE,
    "pred_drift": OBS_PRED_DRIFT_TABLE,
}

# Model
MODEL_NAME = "UNIBOX_CUSTBPR_WEEKLY_FORECAST"
BKCC = "MXBEB"
TIME_COL = "week"

# Alert levels
ALERT_LEVEL_OK = 0
ALERT_LEVEL_WARNING = 1
ALERT_LEVEL_CRITICAL = 2

ALERT_LABELS = {
    ALERT_LEVEL_OK: "ok",
    ALERT_LEVEL_WARNING: "warning",
    ALERT_LEVEL_CRITICAL: "critical",
}

# Metrics to monitor per observability source
PERFORMANCE_METRICS = ["rmse", "wape"]
DATA_DRIFT_FEATURE_METRICS = ["jensen-shannon"]
DATA_DRIFT_SEGMENT_METRICS = ["population_stability_index"]
PRED_DRIFT_METRICS = ["jensen-shannon"]

# Aggregated columns to include
AGGREGATED_COLS = ["stats_ntile_group"]

OBS_METRICS = {
    "performance": PERFORMANCE_METRICS,
    "data_drift_features": DATA_DRIFT_FEATURE_METRICS,
    "data_drift_segments": DATA_DRIFT_SEGMENT_METRICS,
    "pred_drift": PRED_DRIFT_METRICS,
}

# Date filter for alerts
# TODO: cambiar como se maneja el rango de fechas para presentar (permite ver multiples fechas o la mas reciente)
#       asume que se corre como parte de proceso que generó las medidas de drift
LDTS_AFTER = "2026-03-17"
# from datetime import date, timedelta
# LDTS_AFTER = str(date.today() - timedelta(days=1))

REPORT_ALERT_THRESHOLD = ALERT_LEVEL_CRITICAL

# TODO: Poner los correos que deben recibir el reporte
EMAIL_RECIPIENTS = ["david.s.duncan@snowflake.com"]
# TODO: Verificar el nombre de la integración
NOTIFICATION_INTEGRATION = "ALERTAS_EMAIL_MLOPS"  
QUEUE_NOTIFICATION_INTEGRATION = "ALERTAS_QUEUE_MLOPS"

SOURCE_LABELS = {
    "performance": "Performance",
    "data_drift_features": "Feature Drift",
    "data_drift_segments": "Population Drift",
    "pred_drift": "Prediction Drift",
}

# %% [markdown]
# ### 1B. Functions
#
# `get_alerts` consulta una tabla de observabilidad y filtra por modelo,
# nivel de alerta y opcionalmente por versión del modelo y fecha de carga (LDTS).
# `build_alert_report` consolida las alertas de las tres tablas en un solo DataFrame.
# `build_email_html` genera el cuerpo HTML del reporte de alertas para envío por correo.

# %%
def get_alerts(table_name, model_name, metrics=None, aggregated_cols=None,
               model_version=None,
               min_alert_level=ALERT_LEVEL_WARNING, ldts_after=None):
    """Query an observability table and return rows at or above the given alert level.

    Parameters
    ----------
    table_name : str
        Name of the observability table (e.g. OBS_PERFORMANCE).
    model_name : str
        MODEL_NAME value to filter on.
    metrics : list[str] or None
        If provided, only include rows whose METRIC_COL is in this list.
    aggregated_cols : list[str] or None
        If provided, only include rows whose AGGREGATED_COL is in this list.
    model_version : str or None
        If provided, filter to this MODEL_VERSION.
    min_alert_level : int
        Minimum ALERT_LEVEL to include (default 1 = warning and above).
    ldts_after : str or None
        If provided, only include rows with LDTS > this timestamp.

    Returns
    -------
    snowpark.DataFrame
        Filtered alert rows from the table.
    """
    df = (
        session.table(table_name)
        .filter(F.col("MODEL_NAME") == model_name)
        .filter(F.col("ALERT_LEVEL") >= min_alert_level)
    )

    if metrics is not None:
        df = df.filter(F.col("METRIC_COL").isin(metrics))

    if aggregated_cols is not None:
        df = df.filter(F.col("AGGREGATED_COL").isin(aggregated_cols))

    if model_version is not None:
        df = df.filter(F.col("MODEL_VERSION") == model_version)

    if ldts_after is not None:
        df = df.filter(F.col("LDTS") > F.lit(ldts_after).cast("TIMESTAMP_LTZ"))

    return df

# %%
def build_alert_report(model_name, model_version=None,
                       min_alert_level=ALERT_LEVEL_WARNING, ldts_after=None):
    """Build a consolidated alert report across all three observability tables.

    Queries OBS_PERFORMANCE, OBS_DATA_DRIFT, and OBS_PRED_DRIFT, normalises
    the schema (OBS_PERFORMANCE has METRIC_DRIFT; the others do not), adds a
    SOURCE_TABLE column, and returns a single unioned DataFrame.

    Parameters
    ----------
    model_name : str
        MODEL_NAME value to filter on.
    model_version : str or None
        If provided, filter to this MODEL_VERSION.
    min_alert_level : int
        Minimum ALERT_LEVEL to include (default 1 = warning and above).
    ldts_after : str or None
        If provided, only include rows with LDTS > this timestamp.

    Returns
    -------
    snowpark.DataFrame
        Combined alert rows with an additional SOURCE_TABLE column.
    """
    # Shared columns across all three tables
    shared_cols = [
        "RECORD_ID", "MODEL_NAME", "MODEL_VERSION", "ENTITY_MAP",
        "AGGREGATED_COL", "AGGREGATED_VALUE", "METRIC_COL",
        "METRIC_VALUE", "WARNING_THRESHOLD", "CRITICAL_THRESHOLD",
        "ALERT_LEVEL", "BKCC", "CALMONTH", "LDTS",
    ]

    frames = []

    for label, table_name in OBS_TABLES.items():
        metrics = OBS_METRICS.get(label)
        df = get_alerts(
            table_name, model_name,
            metrics=metrics,
            aggregated_cols=AGGREGATED_COLS,
            model_version=model_version,
            min_alert_level=min_alert_level,
            ldts_after=ldts_after,
        )

        # OBS_PERFORMANCE has METRIC_DRIFT; the other two do not
        if table_name == OBS_PERFORMANCE_TABLE:
            selected = df.select(
                *[F.col(c) for c in shared_cols],
                F.col("METRIC_DRIFT"),
                F.lit(label).alias("SOURCE_TABLE"),
            )
        else:
            selected = df.select(
                *[F.col(c) for c in shared_cols],
                F.lit(None).cast("FLOAT").alias("METRIC_DRIFT"),
                F.lit(label).alias("SOURCE_TABLE"),
            )

        frames.append(selected)

    report = frames[0]
    for f in frames[1:]:
        report = report.union_all_by_name(f)

    return report

# %%
import json
import pandas as pd

def get_record_counts(model_name):
    """Query total record counts per source, model version from the OBS tables."""
    frames = []
    for label, table_name in OBS_TABLES.items():
        metrics = OBS_METRICS.get(label)
        df = session.table(table_name).filter(F.col("MODEL_NAME") == model_name)
        if metrics is not None:
            df = df.filter(F.col("METRIC_COL").isin(metrics))
        if AGGREGATED_COLS is not None:
            df = df.filter(F.col("AGGREGATED_COL").isin(AGGREGATED_COLS))
        counts = (
            df.group_by("MODEL_NAME", "MODEL_VERSION")
            .agg(F.count("*").alias("RECORDS_CHECKED"))
        )
        counts = counts.with_column("SOURCE_TABLE", F.lit(label))
        frames.append(counts.to_pandas())
    return pd.concat(frames, ignore_index=True)

def build_email_html(alerts_df, model_name):
    """Build an HTML email body from the consolidated alert report.

    Parameters
    ----------
    alerts_df : snowpark.DataFrame
        Output of build_alert_report, already filtered to the desired alert level.
    model_name : str
        Model name for the report title.

    Returns
    -------
    str
        HTML string suitable for an email body.
    """
    pdf = (
        alerts_df
        .order_by(F.col("ALERT_LEVEL").desc(), "SOURCE_TABLE", "AGGREGATED_VALUE", "METRIC_COL")
        .to_pandas()
    )

    # Extract feature_name from ENTITY_MAP JSON where present
    def _extract_feature(entity_map):
        if entity_map is None:
            return ""
        try:
            obj = json.loads(entity_map) if isinstance(entity_map, str) else entity_map
            return obj.get("feature_name", "")
        except (json.JSONDecodeError, TypeError):
            return ""

    pdf["FEATURE"] = pdf["ENTITY_MAP"].apply(_extract_feature)

    def _extract_time(entity_map):
        if entity_map is None:
            return ""
        try:
            obj = json.loads(entity_map) if isinstance(entity_map, str) else entity_map
            return str(obj.get(TIME_COL, ""))
        except (json.JSONDecodeError, TypeError):
            return ""

    pdf[TIME_COL.upper()] = pdf["ENTITY_MAP"].apply(_extract_time)

    # Truncate LDTS to date
    pdf["FECHA"] = pdf["LDTS"].dt.date

    # --- Inline style constants ---
    STYLE_H1 = 'style="font-family:Arial,sans-serif;color:#8B0000;font-size:22px"'
    STYLE_H2 = 'style="font-family:Arial,sans-serif;color:#8B0000;margin-top:30px;font-size:18px"'
    STYLE_H3 = 'style="font-family:Arial,sans-serif;color:#555;margin-top:20px;font-size:15px"'
    STYLE_P = 'style="font-family:Arial,sans-serif;font-size:14px;color:#333"'
    STYLE_TABLE = 'style="border-collapse:collapse;margin-bottom:20px;font-family:Arial,sans-serif;font-size:13px"'
    STYLE_TH = 'style="background-color:#8B0000;color:white;padding:8px 12px;border:1px solid #aaa;text-align:center"'
    STYLE_TD = 'style="padding:6px 12px;border:1px solid #ccc;text-align:left"'
    STYLE_TD_NUM = 'style="padding:6px 12px;border:1px solid #ccc;text-align:center"'
    STYLE_RECID = 'style="padding:6px 12px;border:1px solid #ccc;font-size:11px"'

    # --- Title & intro ---
    alert_label = ALERT_LABELS.get(REPORT_ALERT_THRESHOLD, "alert")
    total = len(pdf)
    html = f"""
    <h1 {STYLE_H1}>Reporte de Alertas - {model_name}</h1>
    <p {STYLE_P}>Se reportan las alertas de nivel <b>{alert_label}</b> o superior
    detectadas para el modelo <b>{model_name}</b>.</p>
    <p {STYLE_P}>Total de alertas: <b>{total}</b></p>"""

    # --- Summary table (single flat table) ---
    alert_counts = (
        pdf.groupby(["MODEL_NAME", "MODEL_VERSION", "SOURCE_TABLE"])
        .size().reset_index(name="ALERTAS")
    )
    mv_combos = pdf[["MODEL_NAME", "MODEL_VERSION"]].drop_duplicates()
    if mv_combos.empty:
        mv_combos = pd.DataFrame(columns=["MODEL_NAME", "MODEL_VERSION"])
    source_keys = pd.DataFrame({"SOURCE_TABLE": list(OBS_TABLES.keys())})
    skeleton = mv_combos.merge(source_keys, how="cross")
    summary = skeleton.merge(alert_counts, on=["MODEL_NAME", "MODEL_VERSION", "SOURCE_TABLE"], how="left")
    summary["ALERTAS"] = summary["ALERTAS"].fillna(0).astype(int)

    html += f"""
    <h2 {STYLE_H2}>Resumen</h2>
    <table {STYLE_TABLE}>
      <tr><th {STYLE_TH}>Version</th><th {STYLE_TH}>Fuente</th><th {STYLE_TH}>Alertas</th></tr>"""
    summary = summary.sort_values(["MODEL_NAME", "MODEL_VERSION", "SOURCE_TABLE"])
    for _, row in summary.iterrows():
        source_label = SOURCE_LABELS.get(row["SOURCE_TABLE"], row["SOURCE_TABLE"])
        anchor = f"detail-{row['SOURCE_TABLE']}-{row['MODEL_VERSION']}"
        html += f"""
      <tr><td {STYLE_TD}>{row['MODEL_VERSION']}</td><td {STYLE_TD}><a href="#{anchor}">{source_label}</a></td><td {STYLE_TD_NUM}>{row['ALERTAS']}</td></tr>"""
    html += """
    </table>"""

    # --- Detail tables, one per source, sub-grouped by model/version ---
    for source_key in OBS_TABLES:
        source_label = SOURCE_LABELS.get(source_key, source_key)
        section = pdf[pdf["SOURCE_TABLE"] == source_key]
        if section.empty:
            continue

        has_feature = source_key == "data_drift_features"

        anchor = f"detail-{source_key}"
        html += f"""
    <h2 id="{anchor}" {STYLE_H2}>{source_label}</h2>"""

        for (m_name, m_version), group in section.groupby(["MODEL_NAME", "MODEL_VERSION"]):
            # Sort so rows for the same entity are adjacent, worst entities first.
            # Entity = (group, time_col[, feature]) depending on source.
            entity_cols = ["AGGREGATED_VALUE", TIME_COL.upper()]
            if has_feature:
                entity_cols.append("FEATURE")
            group = group.copy()
            group["_rank"] = group.groupby("METRIC_COL")["METRIC_VALUE"].rank(ascending=False)
            group["_avg_rank"] = group.groupby(entity_cols)["_rank"].transform("mean")
            group = group.sort_values(["_avg_rank", *entity_cols, "METRIC_COL"])
            group = group.drop(columns=["_rank", "_avg_rank"])

            anchor = f"detail-{source_key}-{m_version}"
            html += f"""
    <h3 id="{anchor}" {STYLE_H3}>{m_name} &mdash; {m_version}</h3>
    <table {STYLE_TABLE}>
      <tr>
        <th {STYLE_TH}>Grupo</th>
        <th {STYLE_TH}>{TIME_COL.upper()}</th>"""
            if has_feature:
                html += f'<th {STYLE_TH}>Feature</th>'
            html += f"""<th {STYLE_TH}>Metrica</th><th {STYLE_TH}>Valor</th>
        <th {STYLE_TH}>Fecha</th><th {STYLE_TH}>Record ID</th>
      </tr>"""

            for _, row in group.iterrows():
                html += f"""
      <tr>
        <td {STYLE_TD}>{row['AGGREGATED_VALUE']}</td>
        <td {STYLE_TD}>{row[TIME_COL.upper()]}</td>"""
                if has_feature:
                    html += f"<td {STYLE_TD}>{row['FEATURE']}</td>"
                html += f"""
        <td {STYLE_TD}>{row['METRIC_COL']}</td>
        <td {STYLE_TD}>{row['METRIC_VALUE']:.4f}</td>
        <td {STYLE_TD}>{row['FECHA']}</td>
        <td {STYLE_RECID}>{row['RECORD_ID']}</td>
      </tr>"""

            html += """
    </table>"""

    return html

# %%
def build_queue_payload(alerts_df, model_name):
    """Build a JSON payload from the consolidated alert report for queue notification.

    Parameters
    ----------
    alerts_df : snowpark.DataFrame
        Output of build_alert_report, already filtered to the desired alert level.
    model_name : str
        Model name for the report title.

    Returns
    -------
    str
        JSON string suitable for SNOWFLAKE.NOTIFICATION.APPLICATION_JSON.
    """
    pdf = (
        alerts_df
        .order_by(F.col("ALERT_LEVEL").desc(), "SOURCE_TABLE", "AGGREGATED_VALUE", "METRIC_COL")
        .to_pandas()
    )

    def _parse_entity_map(entity_map):
        if entity_map is None:
            return {}
        try:
            return json.loads(entity_map) if isinstance(entity_map, str) else entity_map
        except (json.JSONDecodeError, TypeError):
            return {}

    alert_label = ALERT_LABELS.get(REPORT_ALERT_THRESHOLD, "alert")

    records = []
    for _, row in pdf.iterrows():
        entity = _parse_entity_map(row["ENTITY_MAP"])
        record = {
            "record_id": row["RECORD_ID"],
            "model_name": row["MODEL_NAME"],
            "model_version": row["MODEL_VERSION"],
            "source": row["SOURCE_TABLE"],
            "aggregated_col": row["AGGREGATED_COL"],
            "aggregated_value": row["AGGREGATED_VALUE"],
            TIME_COL: str(entity.get(TIME_COL, "")),
            "metric_col": row["METRIC_COL"],
            "metric_value": float(row["METRIC_VALUE"]) if pd.notna(row["METRIC_VALUE"]) else None,
            "alert_level": int(row["ALERT_LEVEL"]),
            "alert_label": ALERT_LABELS.get(int(row["ALERT_LEVEL"]), "unknown"),
            "calmonth": row.get("CALMONTH", ""),
            "ldts": str(row["LDTS"]),
        }
        feature = entity.get("feature_name")
        if feature:
            record["feature_name"] = feature
        if pd.notna(row.get("METRIC_DRIFT")):
            record["metric_drift"] = float(row["METRIC_DRIFT"])
        records.append(record)

    payload = {
        "model_name": model_name,
        "alert_threshold": alert_label,
        "total_alerts": len(records),
        "bkcc": BKCC,
        "alerts": records,
    }

    return json.dumps(payload)

# %% [markdown]
# ## 2. Run report
#
# Ejecutar la consulta consolidada, generar el HTML del reporte
# y mostrarlo en el notebook.

# %%
alerts_df = build_alert_report(MODEL_NAME, min_alert_level=REPORT_ALERT_THRESHOLD, ldts_after=LDTS_AFTER)

alert_count = alerts_df.count()
print(f"Total alertas (nivel >= {ALERT_LABELS.get(REPORT_ALERT_THRESHOLD, REPORT_ALERT_THRESHOLD)}): {alert_count}")

# %%
email_html = build_email_html(alerts_df, MODEL_NAME)

# TODO: Esta seccion se puede eliminar ya que exista la integración
print(email_html)

# %% [markdown]
# ## 3. Enviar reporte por correo
#
# Pendiente: habilitar una vez que exista la notification integration en la cuenta.

# %%
# TODO: Habilitar cuando la notification integration esté configurada.
#
# recipients_sql = ", ".join(f"'{e}'" for e in EMAIL_RECIPIENTS)
# subject = f"Reporte de Alertas - {MODEL_NAME}"
# safe_html = email_html.replace("'", "''")
#
# session.sql(f"""
#     CALL SYSTEM$SEND_SNOWFLAKE_NOTIFICATION(
#         SNOWFLAKE.NOTIFICATION.TEXT_HTML('{safe_html}'),
#         SNOWFLAKE.NOTIFICATION.EMAIL_INTEGRATION_CONFIG(
#             '{NOTIFICATION_INTEGRATION}',
#             '{subject}',
#             ARRAY_CONSTRUCT({recipients_sql})
#         )
#     )
# """).collect()
#
# print(f"Reporte enviado a: {EMAIL_RECIPIENTS}")

# %% [markdown]
# ## 4. Publicar a Azure Event Grid
#
# Publica el reporte de alertas como mensaje JSON a una cola de Azure Event Grid
# mediante `SYSTEM$SEND_SNOWFLAKE_NOTIFICATION` con `APPLICATION_JSON`.
# Pendiente: habilitar una vez que exista la notification integration en la cuenta.

# %%
queue_payload = build_queue_payload(alerts_df, MODEL_NAME)

# TODO: Esta seccion se puede eliminar ya que exista la integración
print(queue_payload)

# %%
# TODO: Habilitar cuando la notification integration esté configurada.
#
# safe_payload = queue_payload.replace("'", "''")
#
# session.sql(f"""
#     CALL SYSTEM$SEND_SNOWFLAKE_NOTIFICATION(
#         SNOWFLAKE.NOTIFICATION.APPLICATION_JSON('{safe_payload}'),
#         SNOWFLAKE.NOTIFICATION.INTEGRATION('{QUEUE_NOTIFICATION_INTEGRATION}')
#     )
# """).collect()
#
# print(f"Reporte publicado a cola: {QUEUE_NOTIFICATION_INTEGRATION}")

