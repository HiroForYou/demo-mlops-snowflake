# %% [markdown]
# # 06a — Setup & Inference on training data
#
# Part 1 of the baseline generation pipeline.
# Initialises the Snowpark session, creates auxiliary objects and target tables,
# and runs partitioned inference on the training data.
# Results are stored in DA_PREDICTIONS_BASELINE and DA_PREDICTIONS_BASELINE_VW,
# which are required by the subsequent subscripts (06b, 06c, 06d).

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
# ## 3. Create prediction baseline view
#
# Creates a transient table for prediction baselines joined with categories.
# Required by 06b, 06c, and 06d.

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
