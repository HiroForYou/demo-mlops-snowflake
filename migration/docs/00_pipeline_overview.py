# %% [markdown]
# # Pipeline MLOps — Visión Panorámica (NO FUNCIONAL)
#
# Este script NO está diseñado para ejecutarse. Es una referencia visual
# que muestra las secciones clave de cada uno de los 16 scripts del pipeline,
# con el objetivo de dar un entendimiento panorámico del flujo completo.
#
# Pipeline:
#   01 → Data Validation & Cleaning
#   02 → Feature Materialization
#   03 → HPO (Random Search)
#   03b → HPO (Bayesian Search)
#   04 → Many Model Training (16 modelos)
#   05 → Partitioned Model (CustomModel)
#   06a → Setup & Inference Baseline
#   06b → Data Drift Baseline
#   06c → Prediction Drift Baseline
#   06d → Performance Baseline
#   07 → Environment Change (DEV → PROD)
#   08 → Partitioned Inference Batch (producción)
#   09a → Setup Observability
#   09b → Data Drift Detection
#   09c → Prediction Drift Detection
#   09d → Performance Drift Detection

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTES GLOBALES (compartidas entre scripts)
# ═══════════════════════════════════════════════════════════════════════

DATABASE = "BD_AA_DEV"
STORAGE_SCHEMA = "SC_STORAGE_BMX_PS"
FEATURES_SCHEMA = "SC_FEATURES_BMX"
MODELS_SCHEMA = "SC_MODELS_BMX"

MODEL_NAME = "UNI_BOX_REGRESSION_PARTITIONED"
TARGET_COLUMN = "UNI_BOX_WEEK"
PREDICTION_COL = "PREDICTED_UNI_BOX_WEEK"
PARTITION_COL = "STATS_NTILE_GROUP"
TIME_COL = "week"

ID_COLS = ["customer_id", "brand_pres_ret", "prod_key"]
AGG_COLS = ["STATS_NTILE_GROUP", "CUST_CATEGORY"]

PREDICT_INPUT_COLS = [
    "CUSTOMER_ID", "STATS_NTILE_GROUP", "WEEK",
    "BRAND_PRES_RET", "PROD_KEY",
    "SUM_PAST_12_WEEKS", "AVG_PAST_12_WEEKS", "MAX_PAST_24_WEEKS",
    "SUM_PAST_24_WEEKS", "WEEK_OF_YEAR", "AVG_AVG_DAILY_ALL_HOURS",
    "SUM_P4W", "AVG_PAST_24_WEEKS", "PHARM_SUPER_CONV", "WINES_LIQUOR",
    "GROCERIES", "MAX_PREV2", "AVG_PREV2", "MAX_PREV3", "AVG_PREV3",
    "W_M1_TOTAL", "W_M2_TOTAL", "W_M3_TOTAL", "W_M4_TOTAL",
    "SPEC_FOODS", "NUM_COOLERS", "NUM_DOORS",
    "MAX_PAST_4_WEEKS", "SUM_PAST_4_WEEKS", "AVG_PAST_4_WEEKS",
    "MAX_PAST_12_WEEKS",
]

# Mapping grupo → algoritmo (16 grupos)
GROUP_MODEL = {
    "group_stat_0_1": "LGBMRegressor", "group_stat_0_2": "LGBMRegressor",
    "group_stat_0_3": "LGBMRegressor", "group_stat_0_4": "LGBMRegressor",
    "group_stat_1_1": "LGBMRegressor", "group_stat_1_2": "LGBMRegressor",
    "group_stat_1_3": "XGBRegressor",  "group_stat_1_4": "XGBRegressor",
    "group_stat_2_1": "LGBMRegressor", "group_stat_2_2": "LGBMRegressor",
    "group_stat_2_3": "XGBRegressor",  "group_stat_2_4": "XGBRegressor",
    "group_stat_3_1": "LGBMRegressor", "group_stat_3_2": "LGBMRegressor",
    "group_stat_3_3": "LGBMRegressor", "group_stat_3_4": "XGBRegressor",
}


# ═══════════════════════════════════════════════════════════════════════
# 01 — DATA VALIDATION & CLEANING
# ═══════════════════════════════════════════════════════════════════════
# Objetivo: Validar estructura, calidad de datos, y crear datasets limpios
#           con split temporal 90/10 (train/holdout).

# --- Validación ---
train_df = session.table("TRAIN_DATASET_STRUCTURED")
total_rows = train_df.count()
# Verifica NULLs, outliers > 3 std dev, compatibilidad de features

# --- Limpieza y split temporal ---
# Calcula el punto de corte temporal por grupo (90% más antiguo = train)
"""
CREATE TEMPORARY TABLE thresholds AS
WITH cumulative_counts AS (
    SELECT STATS_NTILE_GROUP, WEEK, COUNT(*) as weekly_rows,
           SUM(weekly_rows) OVER (PARTITION BY STATS_NTILE_GROUP ORDER BY WEEK) as running_total,
           SUM(weekly_rows) OVER (PARTITION BY STATS_NTILE_GROUP) as total
    FROM TRAIN_DATASET_STRUCTURED WHERE UNI_BOX_WEEK IS NOT NULL AND UNI_BOX_WEEK >= 0
      AND UNI_BOX_WEEK <= (SELECT PERCENTILE_CONT(0.99) ... )
    GROUP BY STATS_NTILE_GROUP, WEEK
)
SELECT STATS_NTILE_GROUP, MIN(WEEK) as cutoff_week
FROM ... WHERE time_percentile >= 0.90
"""

# TRAIN_DATASET_CLEANED  → WEEK <= cutoff_week  (90% antiguo)
# TRAIN_DATASET_HOLDOUT  → WEEK > cutoff_week   (10% reciente)
# INFERENCE_DATASET_CLEANED → sin NULLs en CUSTOMER_ID/WEEK


# ═══════════════════════════════════════════════════════════════════════
# 02 — FEATURE MATERIALIZATION
# ═══════════════════════════════════════════════════════════════════════
# Objetivo: Crear tabla de features numéricas sin metadata/target.

columns_info = session.sql("DESCRIBE TABLE TRAIN_DATASET_CLEANED").collect()
feature_columns = [col for col in all_columns if col not in EXCLUDED_COLS]

feature_df.write.mode("overwrite").save_as_table("UNI_BOX_FEATURES")


# ═══════════════════════════════════════════════════════════════════════
# 03 / 03b — HYPERPARAMETER SEARCH (Random / Bayesian)
# ═══════════════════════════════════════════════════════════════════════
# Objetivo: Buscar hiperparámetros óptimos por grupo usando Ray cluster.
#           03 usa RandomSearch, 03b usa BayesOpt.

# --- Espacios de búsqueda ---
SEARCH_SPACES = {
    "XGBRegressor": {
        "n_estimators": randint(50, 300),  # BayesOpt: uniform(50, 300) + cast int
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.3),
        # ... subsample, colsample_bytree, gamma, reg_alpha, reg_lambda
    },
    "LGBMRegressor": {
        "n_estimators": randint(50, 300),
        "num_leaves": randint(20, 150),
        # ... learning_rate, subsample, reg_alpha, reg_lambda, min_child_samples
    },
}

# --- Flujo por grupo ---
for group_name in groups_list:  # 16 grupos
    group_df = train_df.filter(STATS_NTILE_GROUP == group_name)
    # Sampling al 20%
    # Split temporal 80/20
    X_train, X_val, y_train, y_val = temporal_train_val_split(...)

    # Tuner de Snowflake ML
    tuner = Tuner(train_func, SEARCH_SPACES[model_type], TunerConfig(
        metric="rmse", mode="min",
        search_alg=RandomSearch(),  # o BayesOpt(utility_kwargs={"kind": "ucb"})
        num_trials=15, max_concurrent_trials=4,
    ))
    results = tuner.run(dataset_map={"train": train_dc, "test": val_dc})

    # Guardar en ML Experiments o tabla HYPERPARAMETER_RESULTS


# ═══════════════════════════════════════════════════════════════════════
# 04 — MANY MODEL TRAINING (16 modelos)
# ═══════════════════════════════════════════════════════════════════════
# Objetivo: Entrenar 16 modelos (uno por grupo STATS_NTILE_GROUP)
#           usando ManyModelTraining distribuido.

# --- Carga de hiperparámetros ---
# Intenta ML Experiments → fallback a tabla HYPERPARAMETER_RESULTS
hyperparams_by_group = {}  # {group_name: {params, val_rmse, algorithm}}

# --- Función de entrenamiento por partición ---
def train_segment_model(data_connector, context):
    segment = context.partition_id    # e.g. "group_stat_0_1"
    model_type = GROUP_MODEL[segment]  # "LGBMRegressor" o "XGBRegressor"
    raw_params = hyperparams_by_group[segment]["params"]

    model = ModelClass(input_cols=feat_cols, label_cols=[TARGET_COLUMN], **params)
    model.fit(train_dataset)
    # Métricas: RMSE, MAE, WAPE, MAPE
    return model

# --- Ejecución distribuida ---
trainer = ManyModelTraining(train_segment_model, "MMT_MODELS")
training_run = trainer.run(partition_by="STATS_NTILE_GROUP", snowpark_dataframe=training_df)

# --- Registro en Model Registry ---
for pid in done_partitions:
    model = training_run.get_model(pid)
    registry.log_model(model, model_name=f"uni_box_regression_{pid}", version_name=f"v_{date}")
    # SET ALIAS=PRODUCTION


# ═══════════════════════════════════════════════════════════════════════
# 05 — PARTITIONED MODEL (CustomModel)
# ═══════════════════════════════════════════════════════════════════════
# Objetivo: Combinar 16 modelos en un solo CustomModel particionado
#           que rutee por STATS_NTILE_GROUP.

class PartitionedUniBoxModel(custom_model.CustomModel):
    @custom_model.partitioned_api
    def predict(self, input_df):
        group_name = input_df["STATS_NTILE_GROUP"].iloc[0]
        model = self.context.model_ref(group_name.lower())
        predictions = model.predict(input_df[feature_cols])
        return pd.DataFrame({
            "CUSTOMER_ID": ..., "STATS_NTILE_GROUP": ..., "WEEK": ...,
            "predicted_uni_box_week": predictions,
        })

# Registro como TABLE_FUNCTION
registry.log_model(partitioned_model,
    model_name="UNI_BOX_REGRESSION_PARTITIONED",
    options={"function_type": "TABLE_FUNCTION"})


# ═══════════════════════════════════════════════════════════════════════
# 06a — SETUP & INFERENCE BASELINE
# ═══════════════════════════════════════════════════════════════════════
# Objetivo: Ejecutar inferencia del modelo PRODUCTION sobre datos de holdout
#           para crear baseline de predicciones.

# Crea tablas: DA_PREDICTIONS_BASELINE, histogramas, performance
# Lookup de categoría de cliente (pharm_super_conv, wines_liquor, etc.)
# Vista: TRAIN_DATASET_HOLDOUT_VW con cust_category

# Inferencia particionada batch por semana:
"""
INSERT INTO DA_PREDICTIONS_BASELINE
SELECT
    SHA2(...) AS RECORD_ID,
    OBJECT_CONSTRUCT('customer_id', p.customer_id, 'week', p.week, ...) AS ENTITY_MAP,
    p.predicted_uni_box_week AS PREDICTION
FROM BATCH_PAGE t,
TABLE(MODEL(UNI_BOX_REGRESSION_PARTITIONED, version)!PREDICT(
    t.CUSTOMER_ID, t.STATS_NTILE_GROUP, ...
) OVER (PARTITION BY t.STATS_NTILE_GROUP)) p
"""

# Crea DA_PREDICTIONS_BASELINE_VW con categorías de cliente


# ═══════════════════════════════════════════════════════════════════════
# 06b — DATA DRIFT BASELINE
# ═══════════════════════════════════════════════════════════════════════
# Objetivo: Crear distribuciones de referencia para detectar drift en features.

# --- PSI (Population Stability Index) ---
# Proporciones de clientes por segmento (STATS_NTILE_GROUP, CUST_CATEGORY)
proportions = (src_tbl
    .group_by(TIME_COL, agg_col)
    .agg(count().alias("N_ROWS"))
    .with_column("PROPORTION", N_ROWS / sum(N_ROWS).over(partition_by(TIME_COL)))
)
# → DA_DATA_DRIFT_HISTOGRAMS_BASELINE (metric_col = "population_stability_index")

# --- JSD (Jensen-Shannon Distance) ---
# Histogramas por feature numérica: 20 cuantiles para high-cardinality,
# midpoints para low-cardinality. Excluye dummies (≤2 valores distintos).
bins, features = get_feature_bin_edges(src_tbl, agg_col)

# Unpivot → join bins → count per bin → scaffold empty bins
# → DA_DATA_DRIFT_HISTOGRAMS_BASELINE (metric_col = "jensen-shannon")


# ═══════════════════════════════════════════════════════════════════════
# 06c — PREDICTION DRIFT BASELINE
# ═══════════════════════════════════════════════════════════════════════
# Objetivo: Histogramas de referencia de la distribución de predicciones.

bins = get_prediction_bin_edges(src_tbl, agg_col)  # 20 cuantiles sobre PREDICTION
# Binning → scaffold → DA_PREDICTION_DRIFT_HISTOGRAMS_BASELINE


# ═══════════════════════════════════════════════════════════════════════
# 06d — PERFORMANCE BASELINE
# ═══════════════════════════════════════════════════════════════════════
# Objetivo: Métricas de rendimiento de referencia comparando predicciones vs actuals.

def compute_performance_metrics(paired_df, agg_col):
    # WAPE = sum|error| / sum|actual|
    # RMSE = sqrt(avg(error²))
    # MAE  = avg|error|
    # F1   = 2 * precision * recall / (precision + recall)
    # Calcula por segmento + "full_model"
    ...

# Join DA_PREDICTIONS_BASELINE_VW con TRAIN_DATASET_HOLDOUT_VW
# → DA_PERFORMANCE_BASELINE


# ═══════════════════════════════════════════════════════════════════════
# 07 — ENVIRONMENT CHANGE (DEV → PROD)
# ═══════════════════════════════════════════════════════════════════════
# Objetivo: Migrar modelo y baselines del schema de desarrollo al de producción.

# Copia versión PRODUCTION al schema target
"""
CREATE MODEL target_model WITH VERSION v_YYYYMMDD
    FROM MODEL source_model VERSION v_YYYYMMDD
-- o --
ALTER MODEL target_model ADD VERSION v_YYYYMMDD
    FROM MODEL source_model VERSION v_YYYYMMDD
"""

# Aplica tag CANDIDATE_VERSION para que script 08 sepa qué versión usar
"""
ALTER MODEL target_model SET TAG CANDIDATE_VERSION = 'v_YYYYMMDD'
"""

# Sincroniza tablas baseline (inserta filas faltantes)
for src_tbl, tgt_tbl in baseline_pairs:
    missing_combos = src.join(tgt, how="left_anti")
    rows_to_insert.write.mode("append").save_as_table(tgt_tbl)


# ═══════════════════════════════════════════════════════════════════════
# 08 — PARTITIONED INFERENCE BATCH (Producción)
# ═══════════════════════════════════════════════════════════════════════
# Objetivo: Ejecutar inferencia en datos de producción para versiones candidatas.

# Lee tags del modelo para obtener versiones candidatas
all_tags = model_ref.show_tags()
versions_to_run = [tag_value for tag in MODEL_TAGS if active]

# Identifica combos (version, week) faltantes en DA_PREDICTIONS
existing_combos = set(
    (row["MODEL_VERSION"], row["ENTITY_TIME"])
    for row in session.table("DA_PREDICTIONS")
        .select("MODEL_VERSION", F.col("ENTITY_MAP")[TIME_COL].alias("ENTITY_TIME"))
        .distinct().collect()
)
combos_needed = {version: [weeks not in existing_combos]}

# Inferencia batch por semana
for version, missing_weeks in combos_needed.items():
    for week in missing_weeks:
        batch_df = features_df.filter(week == tv)
        batch_df.create_or_replace_temp_view("BATCH_PAGE")
        session.sql(insert_batch_sql).collect()  # MODEL()!PREDICT particionado

# Post-inference: crea DA_PREDICTIONS_VW, ACTUALS_TABLE_VW


# ═══════════════════════════════════════════════════════════════════════
# 09a — SETUP OBSERVABILITY
# ═══════════════════════════════════════════════════════════════════════
# Objetivo: Crear las tablas landing para las métricas de observabilidad.

# DA_DATA_DRIFT_HISTOGRAMS, DA_PREDICTION_DRIFT_HISTOGRAMS (mismo schema que baseline)
# DA_DATA_DRIFT, DA_PREDICTION_DRIFT → con WARNING_THRESHOLD, CRITICAL_THRESHOLD, ALERT_LEVEL
# DA_PERFORMANCE → con METRIC_DRIFT, WARNING_THRESHOLD, CRITICAL_THRESHOLD, ALERT_LEVEL


# ═══════════════════════════════════════════════════════════════════════
# 09b — DATA DRIFT DETECTION
# ═══════════════════════════════════════════════════════════════════════
# Objetivo: Detectar cambios en la distribución de datos de entrada.

# --- PSI ---
# Proporciones producción por segmento → compara vs baseline promedio
# psi_component = (inf_prop - base_prop) * ln(inf_prop / base_prop)
# PSI = sum(psi_components)
# Thresholds: > 0.1 warning, > 0.2 critical → DA_DATA_DRIFT

# --- JSD por feature ---
# Reutiliza bin edges del baseline → histogramas de producción
# P = baseline probs, Q = inference probs, M = (P+Q)/2
# JSD = sqrt(0.5 * KL(P||M) + 0.5 * KL(Q||M))
# Thresholds: > 0.2 warning, > 0.45 critical → DA_DATA_DRIFT


# ═══════════════════════════════════════════════════════════════════════
# 09c — PREDICTION DRIFT DETECTION
# ═══════════════════════════════════════════════════════════════════════
# Objetivo: Detectar cambios en la distribución de predicciones.

# Reutiliza bin edges de DA_PREDICTION_DRIFT_HISTOGRAMS_BASELINE
# Histogramas producción → JSD vs baseline
# Thresholds: > 0.2 warning, > 0.45 critical → DA_PREDICTION_DRIFT


# ═══════════════════════════════════════════════════════════════════════
# 09d — PERFORMANCE DRIFT DETECTION
# ═══════════════════════════════════════════════════════════════════════
# Objetivo: Comparar métricas de rendimiento producción vs baseline.

# Join predicciones (DA_PREDICTIONS_VW) con actuals (ACTUALS_TABLE_VW)
# Calcula WAPE, RMSE, MAE, F1 por segmento + full_model
# Drift proporcional: (producción - baseline) / |baseline|

PERF_THRESHOLDS = {
    "wape":      {"warn": 0.20, "crit": 0.50},   # +20% / +50% = worse
    "rmse":      {"warn": 0.20, "crit": 0.50},
    "mae":       {"warn": 0.20, "crit": 0.50},
    "f1_binary": {"warn": -0.15, "crit": -0.30},  # -15% / -30% = worse
}

# ALERT_LEVEL: 0 = OK, 1 = warning, 2 = critical → DA_PERFORMANCE

# === Verificación final ===
# Conteo de todas las landing tables:
# DA_DATA_DRIFT_HISTOGRAMS, DA_DATA_DRIFT,
# DA_PREDICTION_DRIFT_HISTOGRAMS, DA_PREDICTION_DRIFT,
# DA_PERFORMANCE
