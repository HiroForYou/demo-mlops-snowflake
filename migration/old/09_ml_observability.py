# %% [markdown]
# # ML Observability — Production
#
# This notebook implements continuous model monitoring in production.
# It compares current data and inferences against the baselines generated in notebook 06.
# It calculates three types of metrics: (1) data drift — PSI per segment and Jensen-Shannon per feature,
# (2) prediction drift — Jensen-Shannon on the prediction distribution,
# and (3) performance drift — WAPE, RMSE, MAE, and F1 compared to the baseline.
# Results are stored in landing tables with alert levels (0=OK, 1=warning, 2=critical).

# %% [markdown]
# ## 1. Setup
#
# Initial setup: Snowpark session, constants, landing table creation,
# and auxiliary functions to identify new combinations to process.

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F, Window, Row
session = get_active_session()

# %% [markdown]
# ### 1A. Constants
#
# Project constants: input tables (features, predictions, actuals),
# baseline tables (read-only, populated by notebooks 06+07), landing tables
# (created by this notebook), and drift metrics and thresholds configuration.

# %%
# Account info
DATABASE = "BD_AA_DEV"
STORAGE_SCHEMA = "SC_FEATURES_BMX"
FEATURES_SCHEMA = "SC_FEATURES_BMX"

session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()

# Input data sources (all in SC_FEATURES_BMX)
FEATURE_TABLE = f"{DATABASE}.{FEATURES_SCHEMA}.INFERENCE_DATASET_CLEANED_VW"
PREDICTION_TABLE = "DA_PREDICTIONS_VW"
ACTUALS_TABLE = "ACTUALS_TABLE_VW"

# Baseline tables (read-only, populated by 16+17)
DATA_DRIFT_HISTOGRAMS_BASELINE = f"DA_DATA_DRIFT_HISTOGRAMS_BASELINE"
PRED_DRIFT_HISTOGRAMS_BASELINE = f"DA_PREDICTION_DRIFT_HISTOGRAMS_BASELINE"
PERF_BASELINE = f"DA_PERFORMANCE_BASELINE"

# Landing tables (created by this notebook)
DATA_DRIFT_HISTOGRAMS = f"DA_DATA_DRIFT_HISTOGRAMS"
DATA_DRIFT = f"DA_DATA_DRIFT"
PRED_DRIFT_HISTOGRAMS = f"DA_PREDICTION_DRIFT_HISTOGRAMS"
PRED_DRIFT = f"DA_PREDICTION_DRIFT"
PERF_DRIFT = f"DA_PERFORMANCE"

# Model specifics
MODEL_NAME = "UNI_BOX_REGRESSION_PARTITIONED"
ID_COLS = ["customer_id", "brand_pres_ret", "prod_key"]
AGG_COLS = ["STATS_NTILE_GROUP", "CUST_CATEGORY"]
TARGET_COL = "ACTUAL_SALES"
PREDICTION_COL = "PREDICTED_UNI_BOX_WEEK"
TIME_COL = "week"

NON_FEATURE_COLS = {
    *ID_COLS, TIME_COL, TARGET_COL, *AGG_COLS,
}

NON_DRIFT_COLS = {'WEEK_OF_YEAR'}

# Feature drift settings
N_BINS = 20

# Performance metric settings
PERF_JOIN_KEYS = ["CUSTOMER_ID", "BRAND_PRES_RET", TIME_COL]  # actuals table has no PROD_KEY
PERF_METRIC_NAMES = ["wape", "rmse", "mae", "f1_binary"]

# %% [markdown]
# ### 1B. Create landing tables
#
# Creates landing tables for drift histograms (data and predictions),
# drift metrics (with thresholds and alert levels), and performance metrics.

# %%
HISTOGRAM_SCHEMA = """
    RECORD_ID        VARCHAR(128) NOT NULL,
    MODEL_NAME       VARCHAR(64)  NOT NULL,
    MODEL_VERSION    VARCHAR(32)  NOT NULL,
    ENTITY_MAP       OBJECT,
    AGGREGATED_COL   VARCHAR(64)  NOT NULL,
    AGGREGATED_VALUE VARCHAR(64),
    METRIC_COL       VARCHAR(64)  NOT NULL,
    METRIC_MAP       OBJECT,
    CALMONTH         VARCHAR(6),
    LDTS             TIMESTAMP_LTZ(9) NOT NULL
"""

for tbl in [DATA_DRIFT_HISTOGRAMS, PRED_DRIFT_HISTOGRAMS]:
    session.sql(f"CREATE TABLE IF NOT EXISTS {tbl} ({HISTOGRAM_SCHEMA})").collect()


DRIFT_SCHEMA = """
    RECORD_ID          VARCHAR(128) NOT NULL,
    MODEL_NAME         VARCHAR(64)  NOT NULL,
    MODEL_VERSION      VARCHAR(32)  NOT NULL,
    ENTITY_MAP         OBJECT,
    AGGREGATED_COL     VARCHAR(64)  NOT NULL,
    AGGREGATED_VALUE   VARCHAR(64)  NOT NULL,
    METRIC_COL         VARCHAR(64)  NOT NULL,
    METRIC_VALUE       FLOAT,
    WARNING_THRESHOLD  FLOAT        NOT NULL,
    CRITICAL_THRESHOLD FLOAT        NOT NULL,
    ALERT_LEVEL        INT NOT NULL,
    BKCC               VARCHAR(5)   NOT NULL,
    CALMONTH           VARCHAR(6),
    LDTS               TIMESTAMP_LTZ(9) NOT NULL
"""

for tbl in [DATA_DRIFT, PRED_DRIFT]:
    session.sql(f"CREATE TABLE IF NOT EXISTS {tbl} ({DRIFT_SCHEMA})").collect()


PERF_SCHEMA = """
    RECORD_ID          VARCHAR(128) NOT NULL,
    MODEL_NAME         VARCHAR(64)  NOT NULL,
    MODEL_VERSION      VARCHAR(32)  NOT NULL,
    ENTITY_MAP         OBJECT,
    AGGREGATED_COL     VARCHAR(64)  NOT NULL,
    AGGREGATED_VALUE   VARCHAR(64)  NOT NULL,
    METRIC_COL         VARCHAR(64)  NOT NULL,
    METRIC_VALUE       FLOAT,
    METRIC_DRIFT       FLOAT,
    WARNING_THRESHOLD  FLOAT        NOT NULL,
    CRITICAL_THRESHOLD FLOAT        NOT NULL,
    ALERT_LEVEL        INT NOT NULL,
    BKCC               VARCHAR(5)   NOT NULL,
    CALMONTH           VARCHAR(6),
    LDTS               TIMESTAMP_LTZ(9) NOT NULL
"""

session.sql(f"CREATE TABLE IF NOT EXISTS {PERF_DRIFT} ({PERF_SCHEMA})").collect()

print("Landing tables ready.")

# %% [markdown]
# ### 1C. Functions
#
# Auxiliary function. `get_new_combos` identifies (WEEK, MODEL_VERSION) pairs in the
# predictions table that have not yet been processed in a given landing table,
# avoiding recalculating existing metrics.

# %%
def get_new_combos(landing_table, agg_col, metric_col, min_week=None):
    """Find (WEEK, MODEL_VERSION) pairs in PREDICTION_TABLE not yet in landing_table.

    Parameters
    ----------
    landing_table : str
        Fully-qualified landing table name.
    agg_col : str
        Aggregation column being selected.
    metric_col : str
        Metric being selected.
    min_week : str or None
        If provided, only consider inference weeks >= this value.

    Returns
    -------
    list[Row]
        (WEEK, MODEL_VERSION) pairs needing computation.
    """
    all_combos = (
        session.table(PREDICTION_TABLE)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .select(
            F.col("ENTITY_MAP")["week"].cast("STRING").alias(TIME_COL), # TODO: fix once upstream data are corrected
            F.col("MODEL_VERSION"))
        .distinct()
    )
    if min_week is not None:
        all_combos = all_combos.filter(F.col(TIME_COL) >= min_week)

    # Check landing table for existing combos
    existing_combos = (
        session.table(landing_table)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(F.col("AGGREGATED_COL") == agg_col.lower())
        .filter(F.col("METRIC_COL") == metric_col)
        .select(
            F.col("ENTITY_MAP")[TIME_COL].cast("STRING").alias(TIME_COL),
            F.col("MODEL_VERSION"),
        )
        .distinct()
    )
    new_combos = all_combos.join(
        existing_combos, on=[TIME_COL, "MODEL_VERSION"], how="left_anti"
    )

    new_combos_list = new_combos.collect()

    agg_label = f" [{agg_col}]" if agg_col else ""
    print(f"{landing_table}{agg_label}")
    print(f"  New ({TIME_COL}, MODEL_VERSION) combos: {len(new_combos_list)}")

    return new_combos_list

# %% [markdown]
# ## 2. Data drift
#
# Detects changes in the input data distribution relative to the baseline.
# Two complementary metrics are calculated: PSI (segment composition stability)
# and Jensen-Shannon Distance (individual numerical feature drift).

# %% [markdown]
# ### 2A. Segment stability (PSI)
#
# Calculates the Population Stability Index (PSI) by comparing customer proportions
# per segment in production vs. the baseline. It first generates segment histograms
# and then calculates the aggregate PSI. Thresholds: >0.1 = warning, >0.2 = critical.

# %%
def compute_segment_histograms(combos, target_table, agg_col):
    """Compute per-segment customer proportions from inference data and write to target_table.

    Parameters
    ----------
    combos : list[Row]
        (WEEK, MODEL_VERSION) pairs to compute.
    target_table : str
        Table to write results to.
    agg_col : str
        Column to aggregate by (e.g. "STATS_NTILE_GROUP").
    """
    agg_col_lower = agg_col.lower()
    frames = []

    for combo in combos:
        week, version = combo[0], combo[1]
        src_tbl = session.table(FEATURE_TABLE).filter(F.col(TIME_COL) == week)

        data_date_expr = F.dateadd(
            "week", F.col(TIME_COL) % F.lit(100) - F.lit(1),
            F.date_from_parts(F.floor(F.col(TIME_COL) / F.lit(100)), F.lit(1), F.lit(1)),
        )

        proportions = (
            src_tbl
            .group_by(TIME_COL, agg_col)
            # TODO: review whether count_distinct(CUSTOMER_ID) is more appropriate than row count
            .agg(F.count(F.lit(1)).alias("N_ROWS"))
            .with_column(
                "PROPORTION",
                F.col("N_ROWS")
                / F.sum("N_ROWS").over(Window.partition_by(TIME_COL)),
            )
            .with_column("DATA_DATE", data_date_expr)
            .select(
                F.sha2(
                    F.concat(
                        F.lit(MODEL_NAME), F.lit("||"),
                        F.lit(version), F.lit("||"),
                        F.col(TIME_COL), F.lit("||"),
                        F.lit(agg_col_lower), F.lit("||"),
                        F.col(agg_col).cast("STRING"), F.lit("||"),
                        F.lit("population_stability_index"),
                    ),
                    256,
                ).alias("RECORD_ID"),
                F.lit(MODEL_NAME).alias("MODEL_NAME"),
                F.lit(version).alias("MODEL_VERSION"),
                F.object_construct(
                    F.lit(TIME_COL), F.col(TIME_COL),
                    F.lit("data_date"), F.col("DATA_DATE"),
                ).alias("ENTITY_MAP"),
                F.lit(agg_col_lower).alias("AGGREGATED_COL"),
                F.col(agg_col).cast("STRING").alias("AGGREGATED_VALUE"),
                F.lit("population_stability_index").alias("METRIC_COL"),
                F.object_construct(
                    F.lit("bin_count"), F.col("N_ROWS"),
                    F.lit("bin_prop"), F.col("PROPORTION"),
                ).alias("METRIC_MAP"),
                F.to_char(F.col("DATA_DATE"), F.lit("YYYYMM")).alias("CALMONTH"),
                F.current_timestamp().alias("LDTS"),
            )
        )
        frames.append(proportions)

    result = frames[0]
    for f in frames[1:]:
        result = result.union_all_by_name(f)

    result.write.mode("append").save_as_table(target_table)

    n_rows = session.table(target_table).count()
    print(f"Wrote {len(combos)} combo(s) for {agg_col}. {target_table} now has {n_rows} rows.")

# %%
# Get new (WEEK, MODEL_VERSION) combos for segment histograms

psi_new_combos_list = {}

for agg_col in AGG_COLS:
    psi_new_combos_list[agg_col] = get_new_combos(
        DATA_DRIFT_HISTOGRAMS, agg_col=agg_col,
        metric_col="population_stability_index")

# %%
# Calculate segment proportions — loop over each aggregation column

for agg_col in AGG_COLS:
    if psi_new_combos_list[agg_col]:
        compute_segment_histograms(
            psi_new_combos_list[agg_col],
            target_table=DATA_DRIFT_HISTOGRAMS, agg_col=agg_col)
    else:
        print(f"  No new combos for {agg_col} segment histograms, skipping.")

# %%
# Compute PSI from histograms

EPSILON = F.lit(1e-10)

for agg_col in AGG_COLS:
    if not psi_new_combos_list[agg_col]:
        print(f"  No new combos for {agg_col} PSI, skipping.")
        continue

    combo_keys = [F.lit(f'{row[0]}|{row[1]}') for row in psi_new_combos_list[agg_col]]

    # Baseline: average proportion per segment
    baseline = (
        session.table(DATA_DRIFT_HISTOGRAMS_BASELINE)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(F.col("METRIC_COL") == "population_stability_index")
        .filter(F.col("AGGREGATED_COL") == agg_col.lower())
        .select(
            F.col("AGGREGATED_VALUE"),
            F.col("METRIC_MAP")["bin_prop"].cast("DOUBLE").alias("PROPORTION"),
        )
        .group_by("AGGREGATED_VALUE")
        .agg(F.avg("PROPORTION").alias("BASELINE_PROPORTION"))
    )

    # Inference: only new combos
    inference = (
        session.table(DATA_DRIFT_HISTOGRAMS)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(F.col("METRIC_COL") == "population_stability_index")
        .filter(F.col("AGGREGATED_COL") == agg_col.lower())
        .with_column(TIME_COL, F.col("ENTITY_MAP")[TIME_COL].cast("STRING"))
        .filter(
            F.concat(F.col(TIME_COL), F.lit("|"), F.col("MODEL_VERSION"))
            .isin(combo_keys)
        )
        .select(F.col(TIME_COL), F.col("MODEL_VERSION"),
                F.col("AGGREGATED_VALUE"),
                F.col("METRIC_MAP")["bin_prop"].cast("DOUBLE").alias("INFERENCE_PROPORTION"))
    )

    # PSI per (WEEK, MODEL_VERSION)
    inf_prop = F.greatest(F.col("INFERENCE_PROPORTION"), EPSILON)
    base_prop = F.greatest(F.col("BASELINE_PROPORTION"), EPSILON)
    psi_component = (inf_prop - base_prop) * F.ln(inf_prop / base_prop)

    psi_df = (
        inference.join(baseline, on="AGGREGATED_VALUE")
        .with_column("PSI_COMPONENT", psi_component)
        .group_by(TIME_COL, "MODEL_VERSION")
        .agg(F.sum("PSI_COMPONENT").alias("PSI"))
        .with_column(
            "DATA_DATE",
            F.dateadd("week", F.col(TIME_COL) % F.lit(100) - F.lit(1),
                      F.date_from_parts(F.floor(F.col(TIME_COL) / F.lit(100)), F.lit(1), F.lit(1))),
        )
        .select(
            F.sha2(
                F.concat(
                    F.lit(MODEL_NAME), F.lit("||"),
                    F.col("MODEL_VERSION"), F.lit("||"),
                    F.col(TIME_COL), F.lit("||"),
                    F.lit(agg_col.lower()), F.lit("||"),
                    F.lit("full_model"), F.lit("||"),
                    F.lit("population_stability_index"),
                ),
                256,
            ).alias("RECORD_ID"),
            F.lit(MODEL_NAME).alias("MODEL_NAME"),
            F.col("MODEL_VERSION"),
            F.object_construct(
                F.lit(TIME_COL), F.col(TIME_COL),
                F.lit("data_date"), F.col("DATA_DATE"),
            ).alias("ENTITY_MAP"),
            F.lit(agg_col.lower()).alias("AGGREGATED_COL"),
            F.lit("full_model").alias("AGGREGATED_VALUE"),
            F.lit("population_stability_index").alias("METRIC_COL"),
            F.col("PSI").cast("DOUBLE").alias("METRIC_VALUE"),
            F.lit(0.1).cast("DOUBLE").alias("WARNING_THRESHOLD"),
            F.lit(0.2).cast("DOUBLE").alias("CRITICAL_THRESHOLD"),
            F.when(F.col("PSI") > 0.2, F.lit(2))
             .when(F.col("PSI") > 0.1, F.lit(1))
             .otherwise(F.lit(0)).alias("ALERT_LEVEL"),
            F.lit("MXBEB").alias("BKCC"),
            F.to_char(F.col("DATA_DATE"), F.lit("YYYYMM")).alias("CALMONTH"),
            F.current_timestamp().alias("LDTS"),
        )
    )

    psi_df.write.mode("append").save_as_table(DATA_DRIFT)

    n_rows = session.table(DATA_DRIFT).count()
    print(f"PSI for {agg_col}: DATA_DRIFT now has {n_rows} rows.")

# %% [markdown]
# ### 2B. Feature drift (Jensen-Shannon distance)
#
# Calculates the Jensen-Shannon distance per numerical feature, comparing production
# histograms against baseline ones. Reuses bin edges stored in the baseline
# to ensure consistency. Thresholds: >0.2 = warning, >0.45 = critical.

# %%
def get_feature_bin_edges(model_version, agg_col):
    """Retrieve stored baseline bin edges for feature drift.

    Reads edges from DATA_DRIFT_HISTOGRAMS_BASELINE filtered by model_version.

    Parameters
    ----------
    model_version : str
        Model version string to filter baseline rows.
    agg_col : str
        Column name to aggregate by (e.g. "STATS_NTILE_GROUP").

    Returns
    -------
    tuple[snowpark.DataFrame, list[str]]
        (bins_df, features) where bins_df has columns
        AGGREGATED_VALUE, FEATURE_NAME, BIN, BIN_LOW, BIN_HIGH;
        and features is the list of monitored feature names.
    """
    bins = (
        session.table(DATA_DRIFT_HISTOGRAMS_BASELINE)
        .filter(
            (F.col("MODEL_VERSION") == model_version)
            & (F.col("MODEL_NAME") == MODEL_NAME)
            & (F.col("AGGREGATED_COL") == agg_col.lower())
            & (F.col("METRIC_COL") == "jensen-shannon")
        )
        .select(
            F.col("AGGREGATED_VALUE"),
            F.col("ENTITY_MAP")["feature_name"].cast("STRING").alias("FEATURE_NAME"),
            F.col("METRIC_MAP")["bin_number"].cast("INT").alias("BIN"),
            F.col("METRIC_MAP")["bin_low"].cast("FLOAT").alias("BIN_LOW"),
            F.col("METRIC_MAP")["bin_high"].cast("FLOAT").alias("BIN_HIGH"),
        )
        .distinct()
    )
    features = [
        row["FEATURE_NAME"]
        for row in bins.select("FEATURE_NAME").distinct().collect()
    ]

    return bins, features

# %%
def compute_data_drift_histograms(combos, target_table, agg_col):
    """Compute per-feature, per-segment inference histograms using baseline bin edges.

    Parameters
    ----------
    combos : list[Row]
        (WEEK, MODEL_VERSION) pairs to compute.
    target_table : str
        Table to write results to.
    agg_col : str
        Column name to aggregate by (e.g. "STATS_NTILE_GROUP").
    """
    agg_col_lower = agg_col.lower()
    frames = []

    for combo in combos:
        week, version = combo[0], combo[1]
        src_tbl = session.table(FEATURE_TABLE).filter(F.col(TIME_COL) == week)

        # Get bin edges from stored baseline
        bins, features = get_feature_bin_edges(version, agg_col)

        # Unpivot source data into (TIME, SEGMENT, FEATURE_NAME, FEATURE_VALUE)
        cast_exprs = [F.col(f).cast("FLOAT").alias(f) for f in features]
        src_casted = src_tbl.select(
            F.col(TIME_COL), F.col(agg_col), *cast_exprs
        )
        unpivoted = src_casted.unpivot(
            "FEATURE_VALUE", "FEATURE_NAME",
            [F.col(feat) for feat in features],
        ).select(
            F.col(TIME_COL),
            F.col(agg_col).cast("STRING").alias("AGGREGATED_VALUE"),
            "FEATURE_NAME", "FEATURE_VALUE",
        ).filter(F.col("FEATURE_VALUE").is_not_null())

        # Count values per bin
        actual_counts = (
            unpivoted
            .join(bins, on=["FEATURE_NAME", "AGGREGATED_VALUE"])
            .filter(
                (F.col("FEATURE_VALUE") >= F.col("BIN_LOW"))
                & (F.col("FEATURE_VALUE") < F.col("BIN_HIGH"))
            )
            .group_by(TIME_COL, "AGGREGATED_VALUE", "FEATURE_NAME", "BIN", "BIN_LOW", "BIN_HIGH")
            .agg(F.count("*").alias("BIN_COUNT"))
        )

        # Scaffold: every (TIME x bin) so empty bins get a row
        distinct_times = src_casted.select(F.col(TIME_COL)).distinct()
        scaffold = distinct_times.cross_join(bins)

        binned = (
            scaffold.join(
                actual_counts,
                on=[TIME_COL, "AGGREGATED_VALUE", "FEATURE_NAME", "BIN", "BIN_LOW", "BIN_HIGH"],
                how="left",
            )
            .with_column("BIN_COUNT", F.coalesce(F.col("BIN_COUNT"), F.lit(0)))
        )

        # Compute DATA_DATE from WEEK
        data_date_expr = F.dateadd(
            "week",
            F.col(TIME_COL) % F.lit(100) - F.lit(1),
            F.date_from_parts(
                F.floor(F.col(TIME_COL) / F.lit(100)),
                F.lit(1), F.lit(1),
            ),
        )

        histograms = (
            binned
            .with_column("DATA_DATE", data_date_expr)
            .select(
                F.sha2(
                    F.concat(
                        F.lit(MODEL_NAME), F.lit("||"),
                        F.lit(version), F.lit("||"),
                        F.col(TIME_COL), F.lit("||"),
                        F.lit(agg_col_lower), F.lit("||"),
                        F.col("AGGREGATED_VALUE").cast("STRING"), F.lit("||"),
                        F.col("FEATURE_NAME"), F.lit("||"),
                        F.col("BIN").cast("STRING"), F.lit("||"),
                        F.lit("jensen-shannon"),
                    ),
                    256,
                ).alias("RECORD_ID"),
                F.lit(MODEL_NAME).alias("MODEL_NAME"),
                F.lit(version).alias("MODEL_VERSION"),
                F.object_construct(
                    F.lit("feature_name"), F.col("FEATURE_NAME"),
                    F.lit(TIME_COL), F.col(TIME_COL),
                    F.lit("data_date"), F.col("DATA_DATE"),
                ).alias("ENTITY_MAP"),
                F.lit(agg_col_lower).alias("AGGREGATED_COL"),
                F.col("AGGREGATED_VALUE"),
                F.lit("jensen-shannon").alias("METRIC_COL"),
                F.object_construct(
                    F.lit("bin_number"), F.col("BIN"),
                    F.lit("bin_low"), F.col("BIN_LOW"),
                    F.lit("bin_high"), F.col("BIN_HIGH"),
                    F.lit("bin_count"), F.col("BIN_COUNT"),
                ).alias("METRIC_MAP"),
                F.to_char(F.col("DATA_DATE"), F.lit("YYYYMM")).alias("CALMONTH"),
                F.current_timestamp().alias("LDTS"),
            )
        )
        frames.append(histograms)

    result = frames[0]
    for f in frames[1:]:
        result = result.union_all_by_name(f)

    result.write.mode("append").save_as_table(target_table)

    n_rows = session.table(target_table).count()
    print(f"Wrote {len(combos)} combo(s) for {agg_col}. {target_table} now has {n_rows} rows.")

# %%
# Get new (WEEK, MODEL_VERSION) combos for feature histograms

jsd_new_combos_list = {}

for agg_col in AGG_COLS:
    jsd_new_combos_list[agg_col] = get_new_combos(
        DATA_DRIFT_HISTOGRAMS, agg_col=agg_col,
        metric_col="jensen-shannon")

# %%
# Calculate feature histograms — loop over each aggregation column

for agg_col in AGG_COLS:
    if jsd_new_combos_list[agg_col]:
        compute_data_drift_histograms(
            jsd_new_combos_list[agg_col],
            target_table=DATA_DRIFT_HISTOGRAMS, agg_col=agg_col)
    else:
        print(f"  No new combos for {agg_col} feature histograms, skipping.")

# %%
EPSILON = F.lit(1e-10)

# Compute Jensen-Shannon Distance from feature histograms
for agg_col in AGG_COLS:
    agg_col_lower = agg_col.lower()

    if not jsd_new_combos_list[agg_col]:
        print(f"  No new combos for {agg_col} JSD, skipping.")
        continue

    combo_keys = [F.lit(f'{row[0]}|{row[1]}') for row in jsd_new_combos_list[agg_col]]

    jsd_common_filters = [
        F.col("METRIC_COL") == "jensen-shannon",
        F.col("AGGREGATED_COL") == agg_col_lower,
    ]

    def _add_jsd_cols(df):
        return (df
            .with_column(TIME_COL, F.col("ENTITY_MAP")[TIME_COL].cast("STRING"))
            .with_column("FEATURE_NAME", F.col("ENTITY_MAP")["feature_name"].cast("STRING"))
            .with_column("BIN", F.col("METRIC_MAP")["bin_number"].cast("INT"))
            .with_column("BIN_COUNT", F.col("METRIC_MAP")["bin_count"].cast("INT"))
        )

    baseline_hist = _add_jsd_cols(
        session.table(DATA_DRIFT_HISTOGRAMS_BASELINE)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(jsd_common_filters[0]).filter(jsd_common_filters[1])
    )
    inference_hist = _add_jsd_cols(
        session.table(DATA_DRIFT_HISTOGRAMS)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(jsd_common_filters[0]).filter(jsd_common_filters[1])
    )

    # Baseline: average bin probability per (SEGMENT, FEATURE_NAME, BIN)
    baseline_total = F.sum("BIN_COUNT").over(
        Window.partition_by(TIME_COL, "AGGREGATED_VALUE", "FEATURE_NAME")
    )
    baseline = (
        baseline_hist
        .with_column("BIN_PROB", F.col("BIN_COUNT") / F.greatest(baseline_total, F.lit(1.0)))
        .group_by("AGGREGATED_VALUE", "FEATURE_NAME", "BIN")
        .agg(F.avg("BIN_PROB").alias("P"))
    )

    # Inference: probabilities for new combos only
    inf_total = F.sum("BIN_COUNT").over(
        Window.partition_by(TIME_COL, "MODEL_VERSION", "AGGREGATED_VALUE", "FEATURE_NAME")
    )
    inference = (
        inference_hist
        .filter(
            F.concat(F.col(TIME_COL), F.lit("|"), F.col("MODEL_VERSION"))
            .isin(combo_keys)
        )
        .with_column("Q", F.col("BIN_COUNT") / inf_total)
        .select(TIME_COL, "MODEL_VERSION", "AGGREGATED_VALUE", "FEATURE_NAME", "BIN",
                F.col("Q"))
    )

    # Full outer join so bins in only one distribution contribute
    joined = inference.join(
        baseline,
        on=["AGGREGATED_VALUE", "FEATURE_NAME", "BIN"],
        how="full",
    )

    p = F.greatest(F.coalesce(F.col("P"), F.lit(0.0)), EPSILON)
    q = F.greatest(F.coalesce(F.col("Q"), F.lit(0.0)), EPSILON)
    m = (p + q) / F.lit(2.0)

    kl_p_m = p * F.ln(p / m)
    kl_q_m = q * F.ln(q / m)

    # JSD = sqrt(0.5 * KL(P||M) + 0.5 * KL(Q||M))
    jsd_df = (
        joined
        .with_column("KL_P_M", kl_p_m)
        .with_column("KL_Q_M", kl_q_m)
        .group_by(TIME_COL, "MODEL_VERSION", "AGGREGATED_VALUE", "FEATURE_NAME")
        .agg(
            F.sum("KL_P_M").alias("SUM_KL_P_M"),
            F.sum("KL_Q_M").alias("SUM_KL_Q_M"),
        )
        .with_column(
            "JS_DISTANCE",
            F.sqrt(F.lit(0.5) * F.col("SUM_KL_P_M") + F.lit(0.5) * F.col("SUM_KL_Q_M")),
        )
        .with_column(
            "DATA_DATE",
            F.dateadd("week", F.col(TIME_COL) % F.lit(100) - F.lit(1),
                      F.date_from_parts(F.floor(F.col(TIME_COL) / F.lit(100)), F.lit(1), F.lit(1))),
        )
        .select(
            F.sha2(
                F.concat(
                    F.lit(MODEL_NAME), F.lit("||"),
                    F.col("MODEL_VERSION"), F.lit("||"),
                    F.col(TIME_COL), F.lit("||"),
                    F.lit(agg_col_lower), F.lit("||"),
                    F.col("AGGREGATED_VALUE").cast("STRING"), F.lit("||"),
                    F.col("FEATURE_NAME"), F.lit("||"),
                    F.lit("jensen-shannon"),
                ),
                256,
            ).alias("RECORD_ID"),
            F.lit(MODEL_NAME).alias("MODEL_NAME"),
            F.col("MODEL_VERSION"),
            F.object_construct(
                F.lit("feature_name"), F.col("FEATURE_NAME"),
                F.lit(TIME_COL), F.col(TIME_COL),
                F.lit("data_date"), F.col("DATA_DATE"),
            ).alias("ENTITY_MAP"),
            F.lit(agg_col_lower).alias("AGGREGATED_COL"),
            F.col("AGGREGATED_VALUE").cast("STRING").alias("AGGREGATED_VALUE"),
            F.lit("jensen-shannon").alias("METRIC_COL"),
            F.col("JS_DISTANCE").cast("DOUBLE").alias("METRIC_VALUE"),
            F.lit(0.2).cast("DOUBLE").alias("WARNING_THRESHOLD"),
            F.lit(0.45).cast("DOUBLE").alias("CRITICAL_THRESHOLD"),
            F.when(F.col("JS_DISTANCE") > 0.45, F.lit(2))
             .when(F.col("JS_DISTANCE") > 0.2, F.lit(1))
             .otherwise(F.lit(0)).alias("ALERT_LEVEL"),
            F.lit("MXBEB").alias("BKCC"),
            F.to_char(F.col("DATA_DATE"), F.lit("YYYYMM")).alias("CALMONTH"),
            F.current_timestamp().alias("LDTS"),
        )
    )

    jsd_df.write.mode("append").save_as_table(DATA_DRIFT)

    n_rows = session.table(DATA_DRIFT).count()
    print(f"JSD for {agg_col}: DATA_DRIFT now has {n_rows} rows.")

# %% [markdown]
# ## 3. Prediction drift
#
# Detects changes in the model prediction distribution relative to the baseline.
# Uses the same Jensen-Shannon methodology as for features, but applied to the
# prediction column. It first generates prediction histograms using baseline bin edges,
# and then calculates the JSD. Thresholds: >0.2 = warning, >0.45 = critical.

# %%
def get_prediction_bin_edges(model_version, agg_col):
    """Retrieve stored baseline bin edges for prediction drift.

    Reads edges from PRED_DRIFT_HISTOGRAMS_BASELINE filtered by model_version.

    Parameters
    ----------
    model_version : str
        Model version string to filter baseline rows.
    agg_col : str
        Column name to aggregate by (e.g. "STATS_NTILE_GROUP").

    Returns
    -------
    snowpark.DataFrame
        bins_df with columns AGGREGATED_VALUE, BIN, BIN_LOW, BIN_HIGH.
    """
    bins = (
        session.table(PRED_DRIFT_HISTOGRAMS_BASELINE)
        .filter(
            (F.col("MODEL_VERSION") == model_version)
            & (F.col("MODEL_NAME") == MODEL_NAME)
            & (F.col("AGGREGATED_COL") == agg_col.lower())
            & (F.col("METRIC_COL") == "jensen-shannon")
        )
        .select(
            F.col("AGGREGATED_VALUE"),
            F.col("METRIC_MAP")["bin_number"].cast("INT").alias("BIN"),
            F.col("METRIC_MAP")["bin_low"].cast("FLOAT").alias("BIN_LOW"),
            F.col("METRIC_MAP")["bin_high"].cast("FLOAT").alias("BIN_HIGH"),
        )
        .distinct()
    )

    return bins

# %%
def compute_prediction_drift_histograms(combos, target_table, agg_col):
    """Compute inference prediction histograms per segment using baseline bin edges.

    Parameters
    ----------
    combos : list[Row]
        (WEEK, MODEL_VERSION) pairs to compute.
    target_table : str
        Table to write results to.
    agg_col : str
        Column name to aggregate by (e.g. "STATS_NTILE_GROUP").
    """
    agg_col_lower = agg_col.lower()
    frames = []

    for combo in combos:
        week, version = combo[0], combo[1]
        src_tbl = (
            session.table(PREDICTION_TABLE)
            .with_column(TIME_COL, F.col("ENTITY_MAP")["week"]) # TODO: fix once upstrea data are corrected
            .filter(F.col("MODEL_NAME") == MODEL_NAME)
            .filter(F.col(TIME_COL) == week)
        )

        # Get bin edges from stored baseline
        bins = get_prediction_bin_edges(version, agg_col)

        # Bin the prediction column
        pred_values = (
            src_tbl
            .select(
                F.col(TIME_COL),
                F.col(agg_col).cast("STRING").alias("AGGREGATED_VALUE"),
                F.col("PREDICTION"),
            )
            .filter(F.col("PREDICTION").is_not_null())
        )

        actual_counts = (
            pred_values
            .join(bins, on=["AGGREGATED_VALUE"])
            .filter(
                (F.col("PREDICTION") >= F.col("BIN_LOW"))
                & (F.col("PREDICTION") < F.col("BIN_HIGH"))
            )
            .group_by(TIME_COL, "AGGREGATED_VALUE", "BIN", "BIN_LOW", "BIN_HIGH")
            .agg(F.count("*").alias("BIN_COUNT"))
        )

        # Scaffold: every (TIME x bin) so empty bins get a row
        distinct_times = src_tbl.select(F.col(TIME_COL)).distinct()
        scaffold = distinct_times.cross_join(bins)

        binned = (
            scaffold.join(
                actual_counts,
                on=[TIME_COL, "AGGREGATED_VALUE", "BIN", "BIN_LOW", "BIN_HIGH"],
                how="left",
            )
            .with_column("BIN_COUNT", F.coalesce(F.col("BIN_COUNT"), F.lit(0)))
        )

        data_date_expr = F.dateadd(
            "week",
            F.col(TIME_COL) % F.lit(100) - F.lit(1),
            F.date_from_parts(
                F.floor(F.col(TIME_COL) / F.lit(100)),
                F.lit(1), F.lit(1),
            ),
        )

        histograms = (
            binned
            .with_column("DATA_DATE", data_date_expr)
            .select(
                F.sha2(
                    F.concat(
                        F.lit(MODEL_NAME), F.lit("||"),
                        F.lit(version), F.lit("||"),
                        F.lit(PREDICTION_COL), F.lit("||"),
                        F.col(TIME_COL), F.lit("||"),
                        F.lit(agg_col_lower), F.lit("||"),
                        F.col("AGGREGATED_VALUE").cast("STRING"), F.lit("||"),
                        F.lit("jensen-shannon"), F.lit("||"),
                        F.col("BIN").cast("STRING"),
                    ),
                    256,
                ).alias("RECORD_ID"),
                F.lit(MODEL_NAME).alias("MODEL_NAME"),
                F.lit(version).alias("MODEL_VERSION"),
                F.object_construct(
                    F.lit("prediction_name"), F.lit(PREDICTION_COL),
                    F.lit(TIME_COL), F.col(TIME_COL),
                    F.lit("data_date"), F.col("DATA_DATE"),
                ).alias("ENTITY_MAP"),
                F.lit(agg_col_lower).alias("AGGREGATED_COL"),
                F.col("AGGREGATED_VALUE"),
                F.lit("jensen-shannon").alias("METRIC_COL"),
                F.object_construct(
                    F.lit("bin_number"), F.col("BIN"),
                    F.lit("bin_low"), F.col("BIN_LOW"),
                    F.lit("bin_high"), F.col("BIN_HIGH"),
                    F.lit("bin_count"), F.col("BIN_COUNT"),
                ).alias("METRIC_MAP"),
                F.to_char(F.col("DATA_DATE"), F.lit("YYYYMM")).alias("CALMONTH"),
                F.current_timestamp().alias("LDTS"),
            )
        )
        frames.append(histograms)

    result = frames[0]
    for f in frames[1:]:
        result = result.union_all_by_name(f)

    result.write.mode("append").save_as_table(target_table)

    n_rows = session.table(target_table).count()
    print(f"Wrote {len(combos)} combo(s) for {agg_col}. {target_table} now has {n_rows} rows.")

# %%
# Get new (WEEK, MODEL_VERSION) combos for prediction histograms

pred_new_combos_list = {}

for agg_col in AGG_COLS:
    pred_new_combos_list[agg_col] = get_new_combos(
        PRED_DRIFT_HISTOGRAMS, agg_col=agg_col,
        metric_col="jensen-shannon")

# %%
# Calculate prediction histograms — loop over each aggregation column

for agg_col in AGG_COLS:
    if pred_new_combos_list[agg_col]:
        compute_prediction_drift_histograms(
            pred_new_combos_list[agg_col],
            target_table=PRED_DRIFT_HISTOGRAMS, agg_col=agg_col)
    else:
        print(f"  No new combos for {agg_col} prediction histograms, skipping.")

# %%
EPSILON = F.lit(1e-10)

# Compute Jensen-Shannon Distance from prediction histograms
for agg_col in AGG_COLS:
    agg_col_lower = agg_col.lower()

    if not pred_new_combos_list[agg_col]:
        print(f"  No new combos for {agg_col} prediction JSD, skipping.")
        continue

    combo_keys = [F.lit(f'{row[0]}|{row[1]}') for row in pred_new_combos_list[agg_col]]

    jsd_common_filters = [
        F.col("METRIC_COL") == "jensen-shannon",
        F.col("AGGREGATED_COL") == agg_col_lower,
    ]

    def _add_jsd_cols(df):
        return (df
            .with_column(TIME_COL, F.col("ENTITY_MAP")[TIME_COL].cast("STRING"))
            .with_column("BIN", F.col("METRIC_MAP")["bin_number"].cast("INT"))
            .with_column("BIN_COUNT", F.col("METRIC_MAP")["bin_count"].cast("INT"))
        )

    baseline_hist = _add_jsd_cols(
        session.table(PRED_DRIFT_HISTOGRAMS_BASELINE)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(jsd_common_filters[0]).filter(jsd_common_filters[1])
    )
    inference_hist = _add_jsd_cols(
        session.table(PRED_DRIFT_HISTOGRAMS)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(jsd_common_filters[0]).filter(jsd_common_filters[1])
    )

    # Baseline: average bin probability per (SEGMENT, BIN)
    baseline_total = F.sum("BIN_COUNT").over(
        Window.partition_by(TIME_COL, "AGGREGATED_VALUE")
    )
    baseline = (
        baseline_hist
        .with_column("BIN_PROB", F.col("BIN_COUNT") / F.greatest(baseline_total, F.lit(1.0)))
        .group_by("AGGREGATED_VALUE", "BIN")
        .agg(F.avg("BIN_PROB").alias("P"))
    )

    # Inference: probabilities for new combos only
    inf_total = F.sum("BIN_COUNT").over(
        Window.partition_by(TIME_COL, "MODEL_VERSION", "AGGREGATED_VALUE")
    )
    inference = (
        inference_hist
        .filter(
            F.concat(F.col(TIME_COL), F.lit("|"), F.col("MODEL_VERSION"))
            .isin(combo_keys)
        )
        .with_column("Q", F.col("BIN_COUNT") / inf_total)
        .select(TIME_COL, "MODEL_VERSION", "AGGREGATED_VALUE", "BIN",
                F.col("Q"))
    )

    # Full outer join so bins in only one distribution contribute
    joined = inference.join(
        baseline,
        on=["AGGREGATED_VALUE", "BIN"],
        how="full",
    )

    p = F.greatest(F.coalesce(F.col("P"), F.lit(0.0)), EPSILON)
    q = F.greatest(F.coalesce(F.col("Q"), F.lit(0.0)), EPSILON)
    m = (p + q) / F.lit(2.0)

    kl_p_m = p * F.ln(p / m)
    kl_q_m = q * F.ln(q / m)

    # JSD = sqrt(0.5 * KL(P||M) + 0.5 * KL(Q||M))
    jsd_df = (
        joined
        .with_column("KL_P_M", kl_p_m)
        .with_column("KL_Q_M", kl_q_m)
        .group_by(TIME_COL, "MODEL_VERSION", "AGGREGATED_VALUE")
        .agg(
            F.sum("KL_P_M").alias("SUM_KL_P_M"),
            F.sum("KL_Q_M").alias("SUM_KL_Q_M"),
        )
        .with_column(
            "JS_DISTANCE",
            F.sqrt(F.lit(0.5) * F.col("SUM_KL_P_M") + F.lit(0.5) * F.col("SUM_KL_Q_M")),
        )
        .with_column(
            "DATA_DATE",
            F.dateadd("week", F.col(TIME_COL) % F.lit(100) - F.lit(1),
                      F.date_from_parts(F.floor(F.col(TIME_COL) / F.lit(100)), F.lit(1), F.lit(1))),
        )
        .select(
            F.sha2(
                F.concat(
                    F.lit(MODEL_NAME), F.lit("||"),
                    F.col("MODEL_VERSION"), F.lit("||"),
                    F.col(TIME_COL), F.lit("||"),
                    F.lit(agg_col_lower), F.lit("||"),
                    F.col("AGGREGATED_VALUE").cast("STRING"),
                    F.lit("jensen-shannon"), 
                ),
                256,
            ).alias("RECORD_ID"),
            F.lit(MODEL_NAME).alias("MODEL_NAME"),
            F.col("MODEL_VERSION"),
            F.object_construct(
                F.lit("prediction_name"), F.lit(PREDICTION_COL),
                F.lit(TIME_COL), F.col(TIME_COL),
                F.lit("data_date"), F.col("DATA_DATE"),
            ).alias("ENTITY_MAP"),
            F.lit(agg_col_lower).alias("AGGREGATED_COL"),
            F.col("AGGREGATED_VALUE").cast("STRING").alias("AGGREGATED_VALUE"),
            F.lit("jensen-shannon").alias("METRIC_COL"),
            F.col("JS_DISTANCE").cast("DOUBLE").alias("METRIC_VALUE"),
            F.lit(0.2).cast("DOUBLE").alias("WARNING_THRESHOLD"),
            F.lit(0.45).cast("DOUBLE").alias("CRITICAL_THRESHOLD"),
            F.when(F.col("JS_DISTANCE") > 0.45, F.lit(2))
             .when(F.col("JS_DISTANCE") > 0.2, F.lit(1))
             .otherwise(F.lit(0)).alias("ALERT_LEVEL"),
            F.lit("MXBEB").alias("BKCC"),
            F.to_char(F.col("DATA_DATE"), F.lit("YYYYMM")).alias("CALMONTH"),
            F.current_timestamp().alias("LDTS"),
        )
    )

    jsd_df.write.mode("append").save_as_table(PRED_DRIFT)

    n_rows = session.table(PRED_DRIFT).count()
    print(f"Prediction JSD for {agg_col}: PRED_DRIFT now has {n_rows} rows.")

# %% [markdown]
# ## 4. Performance drift
#
# Calculates performance metrics (WAPE, RMSE, MAE, F1) on production data
# and compares them against the baseline average to detect degradation.
# Drift is expressed as a proportional change. For error metrics (higher = worse):
# warn at +20%, critical at +50%. For F1 (lower = worse): warn at -15%, critical at -30%.

# %%
def compute_performance_metrics(paired_df, agg_col):
    """Compute WAPE, RMSE, MAE, and F1 from a paired predictions+actuals DataFrame.

    Parameters
    ----------
    paired_df : snowpark.DataFrame
        Must contain columns: WEEK, MODEL_VERSION, TARGET_COL, PREDICTION_COL,
        and the aggregation column *agg_col*.
    agg_col : str
        Column to group by (e.g. "STATS_NTILE_GROUP").

    Returns
    -------
    snowpark.DataFrame
        Long-format with columns: WEEK, MODEL_VERSION, AGGREGATED_COL,
        AGGREGATED_VALUE, METRIC_COL, METRIC_VALUE.
    """
    agg_col_lower = agg_col.lower()
    actual = F.col(TARGET_COL)
    predicted = F.col("PREDICTION")
    abs_error = F.abs(actual - predicted)
    sq_error = F.pow(actual - predicted, F.lit(2))
    actual_positive = F.iff(actual > F.lit(0), F.lit(1), F.lit(0))
    predicted_positive = F.iff(predicted > F.lit(0), F.lit(1), F.lit(0))

    def _metrics_for_group(df, group_cols, agg_value_expr):
        """Compute the 4 metrics for a given grouping."""
        base = df.group_by(*group_cols).agg(
            # WAPE components
            F.sum(abs_error).alias("SUM_ABS_ERROR"),
            F.sum(F.abs(actual)).alias("SUM_ABS_ACTUAL"),
            # RMSE
            F.avg(sq_error).alias("AVG_SQ_ERROR"),
            # MAE
            F.avg(abs_error).alias("MAE_VAL"),
            # F1 components
            F.sum(actual_positive * predicted_positive).alias("TP"),
            F.sum((F.lit(1) - actual_positive) * predicted_positive).alias("FP"),
            F.sum(actual_positive * (F.lit(1) - predicted_positive)).alias("FN"),
        )

        wape = base.select(
            F.col(TIME_COL), F.col("MODEL_VERSION"),
            F.lit(agg_col_lower).alias("AGGREGATED_COL"),
            agg_value_expr.alias("AGGREGATED_VALUE"),
            F.lit("wape").alias("METRIC_COL"),
            (F.col("SUM_ABS_ERROR") / F.greatest(F.col("SUM_ABS_ACTUAL"), F.lit(1e-10)))
                .cast("DOUBLE").alias("METRIC_VALUE"),
        )

        rmse = base.select(
            F.col(TIME_COL), F.col("MODEL_VERSION"),
            F.lit(agg_col_lower).alias("AGGREGATED_COL"),
            agg_value_expr.alias("AGGREGATED_VALUE"),
            F.lit("rmse").alias("METRIC_COL"),
            F.sqrt(F.col("AVG_SQ_ERROR")).cast("DOUBLE").alias("METRIC_VALUE"),
        )

        mae = base.select(
            F.col(TIME_COL), F.col("MODEL_VERSION"),
            F.lit(agg_col_lower).alias("AGGREGATED_COL"),
            agg_value_expr.alias("AGGREGATED_VALUE"),
            F.lit("mae").alias("METRIC_COL"),
            F.col("MAE_VAL").cast("DOUBLE").alias("METRIC_VALUE"),
        )

        precision = F.col("TP") / F.greatest(F.col("TP") + F.col("FP"), F.lit(1e-10))
        recall = F.col("TP") / F.greatest(F.col("TP") + F.col("FN"), F.lit(1e-10))

        f1 = base.select(
            F.col(TIME_COL), F.col("MODEL_VERSION"),
            F.lit(agg_col_lower).alias("AGGREGATED_COL"),
            agg_value_expr.alias("AGGREGATED_VALUE"),
            F.lit("f1_binary").alias("METRIC_COL"),
            (F.lit(2) * precision * recall / F.greatest(precision + recall, F.lit(1e-10)))
                .cast("DOUBLE").alias("METRIC_VALUE"),
        )

        return wape.union_all_by_name(rmse).union_all_by_name(mae).union_all_by_name(f1)

    # Per aggregation value
    per_segment = _metrics_for_group(
        paired_df,
        [TIME_COL, "MODEL_VERSION", agg_col],
        F.col(agg_col).cast("STRING"),
    )

    # Full model (across all aggregation values)
    full_model = _metrics_for_group(
        paired_df,
        [TIME_COL, "MODEL_VERSION"],
        F.lit("full_model"),
    )

    return per_segment.union_all_by_name(full_model)

# %%
# Get weeks where actuals are available (used to filter combos for performance metrics)
actuals_weeks = (
    session.table(ACTUALS_TABLE)
    .select(F.col(TIME_COL))
    .distinct()
    .collect()
)
actuals_week_set = {row[0] for row in actuals_weeks}
print(f"ACTUALS_TABLE has {len(actuals_week_set)} distinct weeks: {sorted(actuals_week_set)}")

# %%
# Get new (WEEK, MODEL_VERSION) combos for performance metrics, filtered to weeks with actuals

# Thresholds for proportional drift (METRIC_DRIFT column)
# For error metrics (higher = worse): warn at +20%, critical at +50%
# For F1 (lower = worse): warn at -15%, critical at -30%
PERF_THRESHOLDS = {
    "wape":      {"warn": 0.20, "crit": 0.50, "direction": "higher_worse"},
    "rmse":      {"warn": 0.20, "crit": 0.50, "direction": "higher_worse"},
    "mae":       {"warn": 0.20, "crit": 0.50, "direction": "higher_worse"},
    "f1_binary": {"warn": -0.15, "crit": -0.30, "direction": "lower_worse"},
}

perf_new_combos_list = {}

for agg_col in AGG_COLS:
    all_combos = get_new_combos(
        PERF_DRIFT, agg_col=agg_col, metric_col=PERF_METRIC_NAMES[0])

    # Filter to only weeks where actuals exist
    perf_new_combos_list[agg_col] = [c for c in all_combos if c[0] in actuals_week_set]
    print(f"  After actuals filter: {len(perf_new_combos_list[agg_col])} combos for {agg_col}")

# %%
# Calculate inference performance metrics and drift — loop over each aggregation column

for agg_col in AGG_COLS:
    perf_combos = perf_new_combos_list[agg_col]

    if not perf_combos:
        print(f"  No new inference combos for {agg_col}, skipping.")
        continue

    # Read baseline averages for this agg_col
    baseline_avg = (
        session.table(PERF_BASELINE)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(F.col("AGGREGATED_COL") == agg_col.lower())
        .group_by("MODEL_VERSION", "AGGREGATED_COL", "AGGREGATED_VALUE", "METRIC_COL")
        .agg(F.avg("METRIC_VALUE").alias("BASELINE_METRIC"))
    )

    frames = []
    for combo in perf_combos:
        week, version = combo[0], combo[1]

        preds = (
            session.table(PREDICTION_TABLE)
            .filter(F.col("MODEL_NAME") == MODEL_NAME)
            .filter(F.col("MODEL_VERSION") == version)
            .with_column("CUSTOMER_ID", F.col("ENTITY_MAP")["customer_id"].cast("STRING"))
            .with_column("BRAND_PRES_RET", F.col("ENTITY_MAP")["brand_pres_ret"].cast("STRING"))
            .with_column("PROD_KEY", F.col("ENTITY_MAP")["prod_key"].cast("STRING"))
            .with_column(TIME_COL, F.col("ENTITY_MAP")["week"].cast("STRING")) # TODO: fix once upstream data are corrected
            .filter(F.col(TIME_COL) == week)
        )
        actuals = (
            session.table(ACTUALS_TABLE)
            .filter(F.col(TIME_COL) == week)
            .drop(AGG_COLS)
        )

        # Left join: predictions drive the set; missing actuals coalesce to 0
        paired = (
            preds.join(actuals, on=PERF_JOIN_KEYS, how="left")
            .with_column(TARGET_COL, F.coalesce(F.col(TARGET_COL), F.lit(0.0)))
        )
        metrics_df = compute_performance_metrics(paired, agg_col)
        frames.append(metrics_df)

    all_metrics = frames[0]
    for f in frames[1:]:
        all_metrics = all_metrics.union_all_by_name(f)

    # Join with baseline averages to compute drift
    with_drift = all_metrics.join(
        baseline_avg,
        on=["MODEL_VERSION", "AGGREGATED_COL", "AGGREGATED_VALUE", "METRIC_COL"],
        how="left",
    ).with_column(
        "METRIC_DRIFT",
        (F.col("METRIC_VALUE") - F.col("BASELINE_METRIC"))
        / F.greatest(F.abs(F.col("BASELINE_METRIC")), F.lit(1e-10)),
    )

    data_date_expr = F.dateadd(
        TIME_COL, F.col(TIME_COL) % F.lit(100) - F.lit(1),
        F.date_from_parts(F.floor(F.col(TIME_COL) / F.lit(100)), F.lit(1), F.lit(1)),
    )

    # Build per-metric threshold and alert columns using CASE expressions
    warn_expr = (
        F.when(F.col("METRIC_COL") == "f1_binary", F.lit(PERF_THRESHOLDS["f1_binary"]["warn"]))
         .otherwise(F.lit(PERF_THRESHOLDS["wape"]["warn"]))
         .cast("DOUBLE")
    )
    crit_expr = (
        F.when(F.col("METRIC_COL") == "f1_binary", F.lit(PERF_THRESHOLDS["f1_binary"]["crit"]))
         .otherwise(F.lit(PERF_THRESHOLDS["wape"]["crit"]))
         .cast("DOUBLE")
    )

    # Alert: for error metrics drift > threshold is bad; for F1 drift < threshold is bad
    alert_expr = (
        F.when(
            F.col("METRIC_COL") == "f1_binary",
            F.when(F.col("METRIC_DRIFT") < F.lit(PERF_THRESHOLDS["f1_binary"]["crit"]), F.lit(2))
             .when(F.col("METRIC_DRIFT") < F.lit(PERF_THRESHOLDS["f1_binary"]["warn"]), F.lit(1))
             .otherwise(F.lit(0))
        ).otherwise(
            F.when(F.col("METRIC_DRIFT") > F.lit(PERF_THRESHOLDS["wape"]["crit"]), F.lit(2))
             .when(F.col("METRIC_DRIFT") > F.lit(PERF_THRESHOLDS["wape"]["warn"]), F.lit(1))
             .otherwise(F.lit(0))
        )
    )

    drift_rows = (
        with_drift
        .with_column("DATA_DATE", data_date_expr)
        .select(
            F.sha2(
                F.concat(
                    F.lit(MODEL_NAME), F.lit("||"),
                    F.col("MODEL_VERSION"), F.lit("||"),
                    F.col(TIME_COL), F.lit("||"),
                    F.col("AGGREGATED_COL"), F.lit("||"),
                    F.col("AGGREGATED_VALUE"), F.lit("||"),
                    F.col("METRIC_COL"),
                ),
                256,
            ).alias("RECORD_ID"),
            F.lit(MODEL_NAME).alias("MODEL_NAME"),
            F.col("MODEL_VERSION"),
            F.object_construct(
                F.lit(TIME_COL), F.col(TIME_COL),
                F.lit("data_date"), F.col("DATA_DATE"),
            ).alias("ENTITY_MAP"),
            F.col("AGGREGATED_COL"),
            F.col("AGGREGATED_VALUE"),
            F.col("METRIC_COL"),
            F.col("METRIC_VALUE").cast("DOUBLE"),
            F.col("METRIC_DRIFT").cast("DOUBLE"),
            warn_expr.alias("WARNING_THRESHOLD"),
            crit_expr.alias("CRITICAL_THRESHOLD"),
            alert_expr.alias("ALERT_LEVEL"),
            F.lit("MXBEB").alias("BKCC"),
            F.to_char(F.col("DATA_DATE"), F.lit("YYYYMM")).alias("CALMONTH"),
            F.current_timestamp().alias("LDTS"),
        )
    )

    drift_rows.write.mode("append").save_as_table(PERF_DRIFT)

    n_rows = session.table(PERF_DRIFT).count()
    print(f"Perf drift for {agg_col}: PERF_DRIFT now has {n_rows} rows.")

# %% [markdown]
# ### 5. Verification summary
# #
# Final summary showing the row count in each landing table to verify
# that all metrics were calculated correctly.

# %%
print("=" * 60)
print("ML Observability — Landing Table Summary")
print("=" * 60)
for tbl_name, tbl_var in [
    ("DA_DATA_DRIFT_HISTOGRAMS", DATA_DRIFT_HISTOGRAMS),
    ("DA_DATA_DRIFT", DATA_DRIFT),
    ("DA_PREDICTION_DRIFT_HISTOGRAMS", PRED_DRIFT_HISTOGRAMS),
    ("DA_PREDICTION_DRIFT", PRED_DRIFT),
    ("DA_PERFORMANCE", PERF_DRIFT),
]:
    n = session.table(tbl_var).count()
    print(f"  {tbl_name:.<45} {n:>8} rows")
print("=" * 60)

