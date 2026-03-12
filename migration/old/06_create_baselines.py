# %% [markdown]
# # Create baselines
#
# This notebook generates the baselines required for model monitoring.
# This step occurs once the model version is finalized.
# Inference is executed on the training data,
# distribution histograms are calculated to detect data and prediction drift,
# and baseline performance metrics (WAPE, RMSE, MAE, F1) are computed.
# The resulting tables serve as a reference against which production data are compared
# in subsequent notebooks (09_ml_observability).

# %% [markdown]
# ## 1. Setup
# #
# Initial setup: Snowpark session, project constants,
# target table creation, and auxiliary functions.

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F, Window
from snowflake.ml.registry import Registry
import time
session = get_active_session()

# %% [markdown]
# ### 1A. Constants
#
# Project constants: database, schemas, input/output tables,
# model specifications (name, ID columns, partition column),
# and configuration for drift calculation and performance metrics.

# %%
# Account information
DATABASE = "BD_AA_DEV"
STORAGE_SCHEMA = "SC_STORAGE_BMX_PS"
FEATURES_SCHEMA = "SC_STORAGE_BMX_PS"
MODELS_SCHEMA = "SC_MODELS_BMX"

session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()

# Input data sources
FEATURE_TABLE = f"{DATABASE}.{FEATURES_SCHEMA}.TRAIN_DATASET_HOLDOUT_VW"

# Target tables (baseline only)
DATA_DRIFT_HISTOGRAMS_BASELINE = "DA_DATA_DRIFT_HISTOGRAMS_BASELINE"
PRED_BASELINE = "DA_PREDICTIONS_BASELINE"
PRED_BASELINE_VW = "DA_PREDICTIONS_BASELINE_VW"
PRED_DRIFT_HISTOGRAMS_BASELINE = "DA_PREDICTION_DRIFT_HISTOGRAMS_BASELINE"
PERF_BASELINE = "DA_PERFORMANCE_BASELINE"

# Auxiliary setup tables/views
TRAIN_DATASET_HOLDOUT = "TRAIN_DATASET_HOLDOUT"
TRAIN_CUST_CATEGORY_LOOKUP = "TRAIN_CUST_CATEGORY_LOOKUP"


# Model specifications
MODEL_NAME = "UNI_BOX_REGRESSION_PARTITIONED"
ID_COLS = ["customer_id", "brand_pres_ret", "prod_key"]
AGG_COLS = ["STATS_NTILE_GROUP", "CUST_CATEGORY"]
PARTITION_COL = "STATS_NTILE_GROUP"
TARGET_COL = "UNI_BOX_WEEK"
PREDICTION_COL = "PREDICTED_UNI_BOX_WEEK"
TIME_COL = "week"

NON_FEATURE_COLS = {
    *ID_COLS, TIME_COL, TARGET_COL, *AGG_COLS,
}

NON_DRIFT_COLS = {'WEEK_OF_YEAR'}

# Inference configuration
MODEL_FQN = f"{DATABASE}.{MODELS_SCHEMA}.{MODEL_NAME}"
PARTITION_COL = "STATS_NTILE_GROUP"
BASELINE_ALIAS = "PRODUCTION"
INFERENCE_SAMPLE_FRACTION = None   # Set e.g. 0.1 to sample 10% per TIME_COL value; None = full dataset

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

# Feature drift configuration
N_BINS = 20

# Performance metrics configuration
PERF_JOIN_KEYS = [*ID_COLS, TIME_COL]
PERF_METRIC_NAMES = ["wape", "rmse", "mae", "f1_binary"]

# %% [markdown]
# ### 1B. Create setup objects
#
# Creates auxiliary lookups and views required for baseline generation.

# %%
session.sql(f"""
CREATE OR REPLACE TRANSIENT TABLE {TRAIN_CUST_CATEGORY_LOOKUP} AS
SELECT DISTINCT
    customer_id,
    brand_pres_ret,
    week,
    stats_ntile_group,
    CASE 
        WHEN pharm_super_conv = 1 THEN 'pharm_super_conv'
        WHEN wines_liquor = 1 THEN 'wines_liquor'
        WHEN groceries = 1 THEN 'groceries'
        WHEN spec_foods = 1 THEN 'spec_foods'
        ELSE 'others'
        END AS cust_category
FROM {TRAIN_DATASET_HOLDOUT}
""").collect()

session.sql(f"""
CREATE OR REPLACE VIEW {FEATURE_TABLE} AS
SELECT vw.*, mp.cust_category
FROM {TRAIN_DATASET_HOLDOUT} AS vw
LEFT JOIN {TRAIN_CUST_CATEGORY_LOOKUP} AS mp
ON vw.customer_id = mp.customer_id
    AND vw.week = mp.week
    AND vw.brand_pres_ret = mp.brand_pres_ret
    AND vw.stats_ntile_group = mp.stats_ntile_group
""").collect()

# %% [markdown]
# ### 1C. Create target tables
#
# Creates the target tables to store drift histograms (data and predictions)
# and baseline performance metrics, if they don't already exist.

# %%
# Create tables with explicit features

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

for tbl in [DATA_DRIFT_HISTOGRAMS_BASELINE, PRED_DRIFT_HISTOGRAMS_BASELINE]:
    session.sql(f"CREATE TABLE IF NOT EXISTS {tbl} ({HISTOGRAM_SCHEMA})").collect()


PERF_SCHEMA = """
    RECORD_ID          VARCHAR(128) NOT NULL,
    MODEL_NAME         VARCHAR(64)  NOT NULL,
    MODEL_VERSION      VARCHAR(32)  NOT NULL,
    ENTITY_MAP         OBJECT,
    AGGREGATED_COL     VARCHAR(64)  NOT NULL,
    AGGREGATED_VALUE   VARCHAR(64)  NOT NULL,
    METRIC_COL         VARCHAR(64)  NOT NULL,
    METRIC_VALUE       FLOAT,
    BKCC               VARCHAR(5)   NOT NULL,
    CALMONTH           VARCHAR(6),
    LDTS               TIMESTAMP_LTZ(9) NOT NULL
"""

session.sql(f"CREATE TABLE IF NOT EXISTS {PERF_BASELINE} ({PERF_SCHEMA})").collect()

PRED_BASELINE_SCHEMA = """
    RECORD_ID        VARCHAR(128) NOT NULL,
    MODEL_NAME       VARCHAR(64)  NOT NULL,
    MODEL_VERSION    VARCHAR(32)  NOT NULL,
    ENTITY_MAP       OBJECT,
    PREDICTION       FLOAT,
    BKCC             VARCHAR(5)   NOT NULL,
    CALMONTH         VARCHAR(6),
    LDTS             TIMESTAMP_LTZ(9) NOT NULL
"""

session.sql(f"CREATE TABLE IF NOT EXISTS {PRED_BASELINE} ({PRED_BASELINE_SCHEMA})").collect()

print("Target tables ready.")

# %% [markdown]
# ### 1C. Functions
#
# Auxiliary functions reused in the following sections.
# `get_new_baseline_versions` identifies model versions that do not yet have
# baselines calculated in a given target table.

# %%
def get_new_baseline_versions(baseline_table, agg_col, metric_col):
    """Find model versions in PRED_BASELINE that are missing from a baseline table.

    Parameters
    ----------
    baseline_table : str
        Fully qualified name of the baseline target table.
    agg_col : str
        Selected aggregation column.
    metric_col : str
        Selected metric.

    Returns
    -------
    list[str]
        Model versions that need baseline computation.
    """
    all_versions = (
        session.table(PRED_BASELINE_VW)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .select("MODEL_VERSION")
        .distinct()
    )

    existing_versions = (
        session.table(baseline_table)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(F.col("AGGREGATED_COL") == agg_col.lower())
        .filter(F.col("METRIC_COL") == metric_col)
        .select("MODEL_VERSION")
        .distinct()
    )

    new_versions = all_versions.join(
        existing_versions, on="MODEL_VERSION", how="left_anti"
    )

    result = [row["MODEL_VERSION"] for row in new_versions.collect()]
    print(f"{baseline_table} [{agg_col}] [{metric_col}]")
    print(f"  Versions needing baseline: {result}")
    return result

# %% [markdown]
# ## 2. Inference on training data
#
# Partitioned inference of the PRODUCTION model is executed on the training data.
# The resulting predictions are stored in DA_PREDICTIONS_BASELINE and serve as
# a reference for calculating prediction histograms and baseline performance metrics.

# %%
registry = Registry(
    session=session,
    database_name=DATABASE,
    schema_name=MODELS_SCHEMA,
)
model_ref = registry.get_model(MODEL_NAME)
baseline_version = model_ref.version(BASELINE_ALIAS)
version = baseline_version.version_name

already_exists = (
    session.table(PRED_BASELINE)
    .filter(F.col("MODEL_NAME") == MODEL_NAME)
    .filter(F.col("MODEL_VERSION") == version)
    .count()
) > 0

needs_baseline = not already_exists
print(f"Alias {BASELINE_ALIAS!r} -> version {version!r}")
print(f"Already exists in {PRED_BASELINE}: {already_exists}")
print(f"Needs baseline: {needs_baseline}")

# %% [markdown]
# ### 2A. Run partitioned inference for new versions
#
# Executes partitioned inference by STATS_NTILE_GROUP using MODEL()!PREDICT
# over each week (WEEK) value. Predictions are inserted into DA_PREDICTIONS_BASELINE
# in batches. Inference is skipped if the version already exists in the table.

# %%

if needs_baseline:
    print(f"Running inference for version: {version}")
    start_time = time.time()

    input_col_refs = ",\n        ".join(f"t.{c}" for c in PREDICT_INPUT_COLS)
    id_hash_parts = " || '||' ||\n            ".join(f"p.{c}" for c in ID_COLS)

    entity_map_keys = (
        [(c, f"p.{c}") for c in ID_COLS]
        + [("partition_col", f"{PARTITION_COL!r}")]
        + [("partition_value", f"p.{PARTITION_COL}")]
        + [("target_col", f"{TARGET_COL!r}")]
        + [(TIME_COL, f"p.{TIME_COL}")]
    )
    entity_map_pairs = ",\n            ".join(
        f"'{k.lower()}', {v}" for k, v in entity_map_keys
    )

    # -- Get distinct TIME_COL values for batch processing -----
    time_values = sorted(
        row[0]
        for row in session.table(FEATURE_TABLE).select(TIME_COL).distinct().collect()
    )
    print(f"  {len(time_values)} {TIME_COL} value(s) to process")

    # -- SQL fragments ----------------------------------------------------
    data_date_sql = (
        "DATEADD('week', p.{time} % 100 - 1, "
        "DATE_FROM_PARTS(FLOOR(p.{time} / 100), 1, 1))"
    ).format(time=TIME_COL)

    entity_map_sql = (
        f"OBJECT_CONSTRUCT(\n"
        f"            {entity_map_pairs},\n"
        f"            'data_date', {data_date_sql}\n"
        f"        )"
    )

    insert_batch_sql = f"""
    INSERT INTO {PRED_BASELINE}
    (RECORD_ID, MODEL_NAME, MODEL_VERSION, ENTITY_MAP,
     PREDICTION, BKCC, CALMONTH, LDTS)
    SELECT
        SHA2(
            {MODEL_NAME!r} || '||' ||
            {version!r} || '||' ||
            p.{TIME_COL} || '||' ||
            {id_hash_parts} || '||' ||
            p.{PARTITION_COL}
        , 256) AS RECORD_ID,
        {MODEL_NAME!r} AS MODEL_NAME,
        {version!r} AS MODEL_VERSION,
        {entity_map_sql} AS ENTITY_MAP,
        p.{PREDICTION_COL} AS PREDICTION,
        'MXBEB' AS BKCC,
        TO_CHAR({data_date_sql}, 'YYYYMM') AS CALMONTH,
        CURRENT_TIMESTAMP() AS LDTS
    FROM BATCH_PAGE t,
    TABLE(
      MODEL({MODEL_FQN}, {version})!PREDICT(
        {input_col_refs}
      ) OVER (PARTITION BY t.{PARTITION_COL})
    ) p
    """

    # -- Execute by TIME_COL value ------------------------------------
    for i, tv in enumerate(time_values):
        batch_df = (
            session.table(FEATURE_TABLE)
            .filter(F.col(TIME_COL) == tv)
        )
        if INFERENCE_SAMPLE_FRACTION is not None and 0 < INFERENCE_SAMPLE_FRACTION < 1:
            batch_df = batch_df.sample(frac=INFERENCE_SAMPLE_FRACTION)

        batch_df.create_or_replace_temp_view("BATCH_PAGE")
        print(f"    {TIME_COL}={tv} ({i+1}/{len(time_values)})")

        session.sql(insert_batch_sql).collect()

    elapsed = time.time() - start_time
    count = (
        session.table(PRED_BASELINE)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(F.col("MODEL_VERSION") == version)
        .count()
    )
    print(f"  {count:,} predictions saved for version {version} in {elapsed:.1f}s")
else:
    print(f"Version {version} already exists in {PRED_BASELINE}, skipping inference.")

# %% [markdown]
# ## 3. Data drift
#
# Calculates the baseline data distribution histograms, required to detect
# drift in production. Includes two metrics: population stability index (PSI) per segment
# and Jensen-Shannon distance (JSD) per numerical feature.

# %% [markdown]
# ### 3A. Population drift
#
# Calculates customer proportions per segment (STATS_NTILE_GROUP, CUST_CATEGORY)
# in the training data. These proportions are used as a reference for the
# PSI (Population Stability Index) calculation in production.

# %%
def compute_segment_histograms(versions, data_source, target_table, agg_col):
    """Compute customer proportions per segment and write to target_table.

    Parameters
    ----------
    versions : list[str]
        Model versions for which to compute baselines.
    data_source : str | dict[str, str]
        Fully qualified table name, or mapping {version: table}.
    target_table : str
        Fully qualified table name to write results to.
    agg_col : str
        Column name to aggregate by (e.g. "STATS_NTILE_GROUP").
    """
    agg_col_lower = agg_col.lower()
    frames = []

    for version in versions:
        src = data_source[version] if isinstance(data_source, dict) else data_source
        src_tbl = session.table(src)

        data_date_expr = F.dateadd(
            "week", F.col(TIME_COL) % F.lit(100) - F.lit(1),
            F.date_from_parts(F.floor(F.col(TIME_COL) / F.lit(100)), F.lit(1), F.lit(1)),
        )

        proportions = (
            src_tbl
            .group_by(TIME_COL, agg_col)
            # TODO: check whether count_distinct(CUSTOMER_ID) is more appropriate than counting rows
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
    print(f"Wrote {len(versions)} version(s) for {agg_col}. {target_table} now has {n_rows} records.")

# %%
# Identifies the model versions that need baseline information and
# generates the baseline histograms

for agg_col in AGG_COLS:
    new_versions = get_new_baseline_versions(
        DATA_DRIFT_HISTOGRAMS_BASELINE, agg_col=agg_col,
        metric_col="population_stability_index")

    if new_versions:
        baseline_source = {v: FEATURE_TABLE for v in new_versions}
        compute_segment_histograms(
            new_versions, baseline_source,
            target_table=DATA_DRIFT_HISTOGRAMS_BASELINE, agg_col=agg_col)

# %% [markdown]
# ### 3B. Feature drift
#
# Generates base histograms per numerical feature and per segment. For each feature,
# bin edges are calculated (quantiles for high cardinality, midpoints for low cardinality)
# and observations are counted per bin. These histograms serve as a reference distribution
# to calculate the Jensen-Shannon distance in production.

# %%
def get_feature_bin_edges(src_tbl, agg_col):
    """Return per-feature per-segment bin edges and the list of monitored features.

    Identifies numeric feature columns from the data source schema
    (excluding non-feature columns), then computes bin edges
    separately for each segment. Features with fewer than N_BINS
    distinct values get one bin per distinct value (edges at midpoints
    between sorted values, bounded by -Inf / +Inf). Features with
    N_BINS or more distinct values use N_BINS quantile-based cut points.

    Parameters
    ----------
    src_tbl : snowpark.DataFrame
        DataFrame pointing to the baseline data.
    agg_col : str
        Column name to aggregate by (e.g. "STATS_NTILE_GROUP").

    Returns
    -------
    tuple[snowpark.DataFrame, list[str]]
        (bins_df, features) where bins_df has columns
        AGGREGATED_VALUE, FEATURE_NAME, BIN, BIN_LOW, BIN_HIGH;
        and features is the list of monitored feature names.
    """
    from snowflake.snowpark.types import (
        DecimalType, DoubleType, FloatType, IntegerType, LongType, ShortType,
    )
    numeric_types = (DecimalType, DoubleType, FloatType, IntegerType, LongType, ShortType)
    features = [
        field.name
        for field in src_tbl.schema.fields
        if isinstance(field.datatype, numeric_types)
        and field.name not in NON_FEATURE_COLS.union(NON_DRIFT_COLS)
    ]

    # Exclude dummy variables (features with <= 2 observed distinct values)
    distinct_counts = (
        src_tbl
        .select([F.count_distinct(F.col(f)).alias(f) for f in features])
        .collect()[0]
    )
    features = [f for f in features if distinct_counts[f] > 2]

    # Split into low cardinality (< N_BINS global distinct values)
    # and high cardinality to only unpivot what each path needs.
    low_card_features = [f for f in features if distinct_counts[f] < N_BINS]
    high_card_features = [f for f in features if distinct_counts[f] >= N_BINS]

    edge_parts = []

    # -- Low cardinality: midpoint edges between consecutive distinct values --
    if low_card_features:
        lc_cast = [F.col(f).cast("FLOAT").alias(f) for f in low_card_features]
        lc_casted = src_tbl.select(F.col(agg_col), *lc_cast)

        lc_unpivoted = (
            lc_casted.unpivot(
                "FEATURE_VALUE", "FEATURE_NAME",
                [F.col(f) for f in low_card_features],
            )
            .select(
                F.col(agg_col).cast("STRING").alias("AGGREGATED_VALUE"),
                "FEATURE_NAME", "FEATURE_VALUE",
            )
            .filter(F.col("FEATURE_VALUE").is_not_null())
        )

        low_card_vals = lc_unpivoted.select(
            "AGGREGATED_VALUE", "FEATURE_NAME", "FEATURE_VALUE"
        ).distinct()

        val_rank = Window.partition_by("AGGREGATED_VALUE", "FEATURE_NAME").order_by("FEATURE_VALUE")
        low_card_ranked = low_card_vals.with_column(
            "VAL_IDX", F.row_number().over(val_rank)
        )

        curr = low_card_ranked.select(
            F.col("AGGREGATED_VALUE"), F.col("FEATURE_NAME"),
            F.col("VAL_IDX"),
            F.col("FEATURE_VALUE").alias("CURR_VAL"),
        )
        nxt = low_card_ranked.select(
            F.col("AGGREGATED_VALUE"), F.col("FEATURE_NAME"),
            (F.col("VAL_IDX") - F.lit(1)).alias("VAL_IDX"),
            F.col("FEATURE_VALUE").alias("NEXT_VAL"),
        )
        midpoints = (
            curr.join(nxt, on=["AGGREGATED_VALUE", "FEATURE_NAME", "VAL_IDX"])
            .select(
                "AGGREGATED_VALUE", "FEATURE_NAME",
                F.col("VAL_IDX").alias("EDGE_IDX"),
                ((F.col("CURR_VAL") + F.col("NEXT_VAL")) / F.lit(2.0))
                    .cast("FLOAT").alias("EDGE_VALUE"),
            )
        )
        edge_parts.append(midpoints)

    # -- High cardinality: quantile-based edges --
    if high_card_features:
        hc_cast = [F.col(f).cast("FLOAT").alias(f) for f in high_card_features]
        hc_casted = src_tbl.select(F.col(agg_col), *hc_cast)

        hc_unpivoted = (
            hc_casted.unpivot(
                "FEATURE_VALUE", "FEATURE_NAME",
                [F.col(f) for f in high_card_features],
            )
            .select(
                F.col(agg_col).cast("STRING").alias("AGGREGATED_VALUE"),
                "FEATURE_NAME", "FEATURE_VALUE",
            )
            .filter(F.col("FEATURE_VALUE").is_not_null())
        )

        quantile_fracs = [i / N_BINS for i in range(1, N_BINS)]
        agg_exprs = [
            F.call_builtin(
                "APPROX_PERCENTILE", F.col("FEATURE_VALUE"), F.lit(q)
            ).alias(f"P{idx}")
            for idx, q in enumerate(quantile_fracs)
        ]

        pct_wide = (
            hc_unpivoted
            .group_by("AGGREGATED_VALUE", "FEATURE_NAME")
            .agg(*agg_exprs)
            .cache_result()
        )

        p_cols = [F.col(f"P{idx}") for idx in range(len(quantile_fracs))]
        quantile_edges = (
            pct_wide
            .unpivot("EDGE_VALUE", "P_LABEL", p_cols)
            .select(
                F.col("AGGREGATED_VALUE"),
                F.col("FEATURE_NAME"),
                (F.call_builtin("REGEXP_SUBSTR", F.col("P_LABEL"), F.lit("\\d+"))
                 .cast("INT") + F.lit(1)).alias("EDGE_IDX"),
                F.col("EDGE_VALUE").cast("FLOAT").alias("EDGE_VALUE"),
            )
        )
        edge_parts.append(quantile_edges)

    # -- Combine interior edges from both paths --
    interior_edges = edge_parts[0]
    for part in edge_parts[1:]:
        interior_edges = interior_edges.union_all_by_name(part)

    # Add sentinel edges: -Inf at index 0, +Inf at (max EDGE_IDX + 1)
    inf_val = float("inf")
    seg_feat = interior_edges.select("AGGREGATED_VALUE", "FEATURE_NAME").distinct()
    max_edge = (
        interior_edges
        .group_by("AGGREGATED_VALUE", "FEATURE_NAME")
        .agg(F.max("EDGE_IDX").alias("MAX_IDX"))
    )

    low_sentinel = seg_feat.select(
        F.col("AGGREGATED_VALUE"), F.col("FEATURE_NAME"),
        F.lit(0).alias("EDGE_IDX"),
        F.lit(-inf_val).cast("FLOAT").alias("EDGE_VALUE"),
    )
    high_sentinel = max_edge.select(
        F.col("AGGREGATED_VALUE"), F.col("FEATURE_NAME"),
        (F.col("MAX_IDX") + F.lit(1)).alias("EDGE_IDX"),
        F.lit(inf_val).cast("FLOAT").alias("EDGE_VALUE"),
    )

    all_edges = low_sentinel.union_all_by_name(
        interior_edges
    ).union_all_by_name(
        high_sentinel
    )

    # Self-join consecutive edges to form (BIN, BIN_LOW, BIN_HIGH)
    low = all_edges.select(
        F.col("AGGREGATED_VALUE"), F.col("FEATURE_NAME"),
        F.col("EDGE_IDX").alias("BIN"),
        F.col("EDGE_VALUE").alias("BIN_LOW"),
    )
    high = all_edges.select(
        F.col("AGGREGATED_VALUE"), F.col("FEATURE_NAME"),
        (F.col("EDGE_IDX") - F.lit(1)).alias("BIN"),
        F.col("EDGE_VALUE").alias("BIN_HIGH"),
    )

    bins = (
        low.join(high, on=["AGGREGATED_VALUE", "FEATURE_NAME", "BIN"])
        .filter(F.col("BIN") >= F.lit(1))
        .select("AGGREGATED_VALUE", "FEATURE_NAME", "BIN", "BIN_LOW", "BIN_HIGH")
    )

    return bins, features

# %%
def compute_data_drift_histograms(versions, data_source, target_table, agg_col):
    """Compute per-feature per-segment baseline histograms and write to target_table.

    Uses bin edges from get_feature_bin_edges computed from the data.
    Each bin gets one record per time period, even when
    there are no values in that bin (BIN_COUNT = 0).

    Parameters
    ----------
    versions : list[str]
        Model versions for which to compute baselines.
    data_source : str | dict[str, str]
        Fully qualified table name, or mapping {version: table}.
    target_table : str
        Fully qualified table name to write results to.
    agg_col : str
        Column name to aggregate by (e.g. "STATS_NTILE_GROUP").
    """
    agg_col_lower = agg_col.lower()
    frames = []

    for version in versions:
        src = data_source[version] if isinstance(data_source, dict) else data_source
        src_tbl = session.table(src)

        # Get bin edges and the list of features to monitor
        bins, features = get_feature_bin_edges(src_tbl, agg_col)

        # Unpivot source data to (TIME, SEGMENT, FEATURE_NAME, FEATURE_VALUE)
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

        # Count values per bin (only bins with data)
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

        # Build scaffold: every (TIME x bin) combination so empty bins get a record
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

        # Build output in long format
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
    print(f"Wrote {len(versions)} version(s) for {agg_col}. {target_table} now has {n_rows} records.")

# %%
# Compute baseline feature histograms -- iterate over each aggregation column

for agg_col in AGG_COLS:
    new_versions = get_new_baseline_versions(
        DATA_DRIFT_HISTOGRAMS_BASELINE, agg_col=agg_col,
        metric_col="jensen-shannon")

    if new_versions:
        baseline_source = {v: FEATURE_TABLE for v in new_versions}
        compute_data_drift_histograms(
            new_versions, baseline_source,
            target_table=DATA_DRIFT_HISTOGRAMS_BASELINE, agg_col=agg_col)

# %% [markdown]
# ## 4. Prediction drift
#
# Calculates baseline histograms of the prediction distribution per segment.
# The same quantile-based bins as for features are used, but applied
# to the prediction column. These histograms serve as a reference to
# detect drift in the model predictions in production.

# %%
def get_prediction_bin_edges(src_tbl, agg_col):
    """Return per-segment bin edges for the prediction column.

    Computes N_BINS quantile-based bin edges from PREDICTION_COL,
    separately for each segment defined by *agg_col*.

    Parameters
    ----------
    src_tbl : snowpark.DataFrame
        DataFrame pointing to the baseline prediction data.
    agg_col : str
        Column name to aggregate by (e.g. "STATS_NTILE_GROUP").

    Returns
    -------
    snowpark.DataFrame
        bins_df with columns AGGREGATED_VALUE, BIN, BIN_LOW, BIN_HIGH.
    """
    quantile_fracs = [i / N_BINS for i in range(1, N_BINS)]
    agg_exprs = [
        F.call_builtin(
            "APPROX_PERCENTILE", F.col("PREDICTION").cast("FLOAT"), F.lit(q)
        ).alias(f"P{idx}")
        for idx, q in enumerate(quantile_fracs)
    ]

    pct_wide = (
        src_tbl
        .group_by(F.col(agg_col).cast("STRING").alias("AGGREGATED_VALUE"))
        .agg(*agg_exprs)
        .cache_result()
    )

    p_cols = [F.col(f"P{idx}") for idx in range(len(quantile_fracs))]
    interior_edges = (
        pct_wide
        .unpivot("EDGE_VALUE", "P_LABEL", p_cols)
        .select(
            F.col("AGGREGATED_VALUE"),
            (F.call_builtin("REGEXP_SUBSTR", F.col("P_LABEL"), F.lit("\\d+"))
             .cast("INT") + F.lit(1)).alias("EDGE_IDX"),
            F.col("EDGE_VALUE").cast("FLOAT").alias("EDGE_VALUE"),
        )
    )

    # Add sentinel edges: -Inf at index 0, +Inf at (max EDGE_IDX + 1)
    inf_val = float("inf")
    segments = interior_edges.select("AGGREGATED_VALUE").distinct()
    max_edge = (
        interior_edges
        .group_by("AGGREGATED_VALUE")
        .agg(F.max("EDGE_IDX").alias("MAX_IDX"))
    )

    low_sentinel = segments.select(
        F.col("AGGREGATED_VALUE"),
        F.lit(0).alias("EDGE_IDX"),
        F.lit(-inf_val).cast("FLOAT").alias("EDGE_VALUE"),
    )
    high_sentinel = max_edge.select(
        F.col("AGGREGATED_VALUE"),
        (F.col("MAX_IDX") + F.lit(1)).alias("EDGE_IDX"),
        F.lit(inf_val).cast("FLOAT").alias("EDGE_VALUE"),
    )

    all_edges = low_sentinel.union_all_by_name(
        interior_edges
    ).union_all_by_name(
        high_sentinel
    )

    # Self-join consecutive edges to form (BIN, BIN_LOW, BIN_HIGH)
    low = all_edges.select(
        F.col("AGGREGATED_VALUE"),
        F.col("EDGE_IDX").alias("BIN"),
        F.col("EDGE_VALUE").alias("BIN_LOW"),
    )
    high = all_edges.select(
        F.col("AGGREGATED_VALUE"),
        (F.col("EDGE_IDX") - F.lit(1)).alias("BIN"),
        F.col("EDGE_VALUE").alias("BIN_HIGH"),
    )

    bins = (
        low.join(high, on=["AGGREGATED_VALUE", "BIN"])
        .filter(F.col("BIN") >= F.lit(1))
        .select("AGGREGATED_VALUE", "BIN", "BIN_LOW", "BIN_HIGH")
    )

    return bins

# %%
def compute_prediction_drift_histograms(versions, data_source, target_table, agg_col):
    """Compute per-segment baseline prediction histograms and write to target_table.

    Parameters
    ----------
    versions : list[str]
        Model versions for which to compute baselines.
    data_source : str | dict[str, str]
        Fully qualified table name, or mapping {version: table}.
    target_table : str
        Fully qualified table name to write results to.
    agg_col : str
        Column name to aggregate by (e.g. "STATS_NTILE_GROUP").
    """
    agg_col_lower = agg_col.lower()
    frames = []

    for version in versions:
        src = data_source[version] if isinstance(data_source, dict) else data_source
        src_tbl = (
            session.table(src)
            .filter(F.col("MODEL_NAME") == MODEL_NAME)
            .with_column(TIME_COL, F.col("ENTITY_MAP")[TIME_COL].cast("NUMBER"))
        )

        # Get bin edges
        bins = get_prediction_bin_edges(src_tbl, agg_col)

        # Assign bins to the prediction column directly (no unpivot)
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

        # Scaffold: every (TIME x bin) so empty bins get a record
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
                        F.col(TIME_COL), F.lit("||"),
                        F.lit(agg_col_lower), F.lit("||"),
                        F.col("AGGREGATED_VALUE").cast("STRING"), F.lit("||"),
                        F.col("BIN").cast("STRING"), F.lit("||"),
                        F.lit("jensen-shannon"),
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
    print(f"Wrote {len(versions)} version(s) for {agg_col}. {target_table} now has {n_rows} records.")

# %%
# Compute baseline prediction histograms -- iterate over each aggregation column

for agg_col in AGG_COLS:
    new_versions = get_new_baseline_versions(
        PRED_DRIFT_HISTOGRAMS_BASELINE, agg_col=agg_col,
        metric_col="jensen-shannon")

    if new_versions:
        baseline_source = {v: PRED_BASELINE_VW for v in new_versions}
        compute_prediction_drift_histograms(
            new_versions, baseline_source,
            target_table=PRED_DRIFT_HISTOGRAMS_BASELINE, agg_col=agg_col)

# %% [markdown]
# ## 5. Performance drift
#
# Calculates the baseline performance metrics (WAPE, RMSE, MAE, F1) by comparing
# baseline predictions with the actual training data.
# These are calculated both per segment and for the complete model ("full_model").
# These metrics serve as a reference to detect performance degradation in production.

# %%
def compute_performance_metrics(paired_df, agg_col):
    """Compute WAPE, RMSE, MAE and F1 from a paired predictions+actuals DataFrame.

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
        Long format with columns: WEEK, MODEL_VERSION, AGGREGATED_COL,
        AGGREGATED_VALUE, METRIC_COL, METRIC_VALUE.
    """
    agg_col_lower = agg_col.lower()
    actual = F.col(TARGET_COL)
    predicted = F.col(PREDICTION_COL)
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
# Compute baseline performance metrics -- iterate over each aggregation column

for agg_col in AGG_COLS:
    new_versions = get_new_baseline_versions(
        PERF_BASELINE, agg_col=agg_col, metric_col=PERF_METRIC_NAMES[0])

    if not new_versions:
        print(f"  No new baseline versions for {agg_col}, skipping.")
        continue

    for version in new_versions:
        # Join baseline predictions with baseline actuals.
        # PRED_BASELINE_VW stores join keys inside ENTITY_MAP, so we extract them.
        preds = (
            session.table(PRED_BASELINE_VW)
            .filter(F.col("MODEL_NAME") == MODEL_NAME)
            .filter(F.col("MODEL_VERSION") == version)
            .with_column("CUSTOMER_ID", F.col("ENTITY_MAP")["customer_id"].cast("STRING"))
            .with_column(TIME_COL, F.col("ENTITY_MAP")[TIME_COL].cast("STRING"))
            .with_column("BRAND_PRES_RET", F.col("ENTITY_MAP")["brand_pres_ret"].cast("STRING"))
            .with_column("PROD_KEY", F.col("ENTITY_MAP")["prod_key"].cast("STRING"))
            .with_column(PREDICTION_COL, F.col("PREDICTION"))
        )
        actuals = session.table(FEATURE_TABLE).drop(AGG_COLS)

        paired = preds.join(actuals, on=PERF_JOIN_KEYS, how="inner")

        # Compute metrics
        metrics_df = compute_performance_metrics(paired, agg_col)

        # Format according to the PERF_BASELINE schema
        data_date_expr = F.dateadd(
            "week", F.col(TIME_COL) % F.lit(100) - F.lit(1),
            F.date_from_parts(F.floor(F.col(TIME_COL) / F.lit(100)), F.lit(1), F.lit(1)),
        )

        baseline_rows = (
            metrics_df
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
                F.col("METRIC_VALUE"),
                F.lit("MXBEB").alias("BKCC"),
                F.to_char(F.col("DATA_DATE"), F.lit("YYYYMM")).alias("CALMONTH"),
                F.current_timestamp().alias("LDTS"),
            )
        )

        baseline_rows.write.mode("append").save_as_table(PERF_BASELINE)

    n_rows = session.table(PERF_BASELINE).count()
    print(f"Baseline performance for {agg_col}: PERF_BASELINE now has {n_rows} records.")

# %% [markdown]
# ## 6. Create prediction baseline view
#
# Creates a transient table for prediction baselines joined with categories.

# %%
session.sql(f"""
CREATE OR REPLACE TRANSIENT TABLE {PRED_BASELINE_VW} AS
SELECT
    vw.*,
    mp.stats_ntile_group,
    mp.cust_category
FROM {PRED_BASELINE} AS vw
LEFT JOIN {TRAIN_CUST_CATEGORY_LOOKUP} AS mp
ON vw.entity_map:customer_id = mp.customer_id
    AND vw.entity_map:brand_pres_ret = mp.brand_pres_ret
    AND vw.entity_map:week = mp.week
    AND vw.entity_map:partition_value = mp.stats_ntile_group
""").collect()

print(f"Created {PRED_BASELINE_VW} table.")

