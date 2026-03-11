# %% [markdown]
# # Partitioned Inference — Inference Dataset
#
# This notebook executes the partitioned inference of the model on production data.
# It reads candidate versions from model tags (CANDIDATE_VERSION), identifies 
# which combinations (version, week) are missing from the predictions table, and executes
# MODEL()!PREDICT partitioned by STATS_NTILE_GROUP for each batch.
# The resulting predictions feed the observability notebook (09).

# %% [markdown]
# ## 1. Setup
#
# Initial setup: Snowpark session, project constants,
# prediction table creation, and inference parameters.

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F
from snowflake.ml.registry import Registry
import time
session = get_active_session()

# %% [markdown]
# ### 1A. Constants
#
# Project constants: database, schemas, inference feature table,
# prediction table (DA_PREDICTIONS), model specifications, input columns
# for PREDICT, and optional sampling and week filter parameters.

# %%
# Account info
DATABASE = "BD_AA_DEV"
STORAGE_SCHEMA = "SC_FEATURES_BMX"
FEATURES_SCHEMA = "SC_FEATURES_BMX"
MODELS_SCHEMA = "SC_STORAGE_BMX_PS"
SRC_STORAGE_SCHEMA = "SC_STORAGE_BMX_PS"

session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()

# Auxiliary setup tables/views
INFERENCE_DATASET_CLEANED = "INFERENCE_DATASET_CLEANED"
INFERENCE_CUST_CATEGORY_LOOKUP = "INFERENCE_CUST_CATEGORY_LOOKUP"
DA_PREDICTIONS_VW = "DA_PREDICTIONS_VW"
GROUND_TRUTH_DATASET_STRUCTURED = "GROUND_TRUTH_DATASET_STRUCTURED"
ACTUALS_TABLE_VW = "ACTUALS_TABLE_VW"

# Input data source (inference dataset)
SOURCE_TABLE = "INFERENCE_DATASET_CLEANED_VW"
FEATURE_TABLE = f"{DATABASE}.{FEATURES_SCHEMA}.INFERENCE_DATASET_CLEANED_VW"

# Landing location
PREDICTION_TABLE = "DA_PREDICTIONS"

# Model specifics
MODEL_NAME = "UNI_BOX_REGRESSION_PARTITIONED"
MODEL_FQN = f"{DATABASE}.{MODELS_SCHEMA}.{MODEL_NAME}"

# Tags whose values identify model versions to run inference for.
# Each tag value is expected to hold a version name (see script 17).
# These are the short tag names as they appear in Model.show_tags().
MODEL_TAGS = ["CANDIDATE_VERSION"]

ID_COLS = ["customer_id", "brand_pres_ret", "prod_key"]
PARTITION_COL = "STATS_NTILE_GROUP"
TIME_COL = "week"
PREDICTION_COL = "PREDICTED_UNI_BOX_WEEK"
TARGET_COL = "UNI_BOX_WEEK"

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

# Inference settings
INFERENCE_SAMPLE_FRACTION = None   # Set to e.g. 0.1 to sample 10% per WEEK; None = full dataset
MIN_INFERENCE_TIME = None          # Set to e.g. "202545" to only process weeks >= this value; None = all weeks

# %% [markdown]
# ### 1B. Create landing table
#
# Creates the DA_PREDICTIONS table if it doesn't exist. Stores individual predictions
# with their RECORD_ID, model version, ENTITY_MAP (record metadata), and prediction.

# %%
PRED_SCHEMA = """
    RECORD_ID        VARCHAR(128) NOT NULL,
    MODEL_NAME       VARCHAR(64)  NOT NULL,
    MODEL_VERSION    VARCHAR(32)  NOT NULL,
    ENTITY_MAP       OBJECT,
    PREDICTION       FLOAT,
    BKCC             VARCHAR(5)   NOT NULL,
    CALMONTH         VARCHAR(6),
    LDTS             TIMESTAMP_LTZ(9) NOT NULL
"""

session.sql(f"CREATE TABLE IF NOT EXISTS {PREDICTION_TABLE} ({PRED_SCHEMA})").collect()
print(f"Landing table ready: {PREDICTION_TABLE}")

# Landing table ready: {PREDICTION_TABLE}

# %% [markdown]
# ### 1C. Create setup objects
#
# Creates auxiliary lookups and views required for inference.

# %%
session.sql(f"""
CREATE OR REPLACE TRANSIENT TABLE {INFERENCE_DATASET_CLEANED}
CLONE {DATABASE}.{SRC_STORAGE_SCHEMA}.{INFERENCE_DATASET_CLEANED}
""").collect()

session.sql(f"""
CREATE OR REPLACE TRANSIENT TABLE {INFERENCE_CUST_CATEGORY_LOOKUP} AS
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
FROM {INFERENCE_DATASET_CLEANED}
""").collect()

session.sql(f"""
CREATE OR REPLACE VIEW {FEATURE_TABLE} AS
SELECT vw.*, mp.cust_category
FROM {INFERENCE_DATASET_CLEANED} AS vw
LEFT JOIN {INFERENCE_CUST_CATEGORY_LOOKUP} AS mp
ON vw.customer_id = mp.customer_id
    AND vw.week = mp.week
    AND vw.brand_pres_ret = mp.brand_pres_ret
""").collect()

# %% [markdown]
# ## 2. Feature creation [PENDIENTE]
#
# Feature creation step (placeholder). Currently passes inference data as-is.
# In the future, it will be replaced with a transformation via Feature Store.

# %%

# TODO: Transform SOURCE_TABLE to FEATURE_TABLE via Feature Store

features_df = session.table(FEATURE_TABLE)

total_rows = features_df.count()
print(f"Inference dataset: {total_rows:,} rows")

# %% [markdown]
# ## 3. Predictions by week
#
# Executes partitioned inference for each candidate model version.
# First resolves versions from tags, identifies which combinations
# (version, week) are missing in DA_PREDICTIONS, and then executes MODEL()!PREDICT
# in batches by week.

# %% [markdown]
# ### 3A. Resolve model versions from tags
#
# Reads model tags (e.g., CANDIDATE_VERSION) and extracts active versions.
# Then identifies which combinations (version, week) already exist in DA_PREDICTIONS
# to avoid recalculating duplicate predictions.

# %%
registry = Registry(
    session=session,
    database_name=DATABASE,
    schema_name=MODELS_SCHEMA,
)
model_ref = registry.get_model(MODEL_NAME)
all_tags = model_ref.show_tags()

# Build a table of tags -> versions and whether they are active (in MODEL_TAGS)
tag_rows = []
for tag_name, tag_value in all_tags.items():
    tag_short = tag_name.split(".")[-1].upper()  # FQN -> short name
    active = tag_short in [t.upper() for t in MODEL_TAGS]
    tag_rows.append({"TAG": tag_short, "VERSION": tag_value, "ACTIVE": active})

tag_df = session.create_dataframe(tag_rows)
tag_df.show()

# Collect the deduplicated set of versions from active tags
versions_to_run = list({r["VERSION"] for r in tag_rows if r["ACTIVE"] and r["VERSION"]})

# %%
# Get all WEEK values from the features data (respecting MIN_INFERENCE_TIME)
all_time_values = sorted(
    row[0]
    for row in features_df.select(TIME_COL).distinct().collect()
)
if MIN_INFERENCE_TIME is not None:
    all_time_values = [tv for tv in all_time_values if tv >= MIN_INFERENCE_TIME]

print(f"{TIME_COL} values to consider: {all_time_values}")

# Get existing (VERSION, WEEK) combos already in PREDICTION_TABLE
existing_combos = set(
    (row["MODEL_VERSION"], row["ENTITY_TIME"])
    for row in (
        session.table(PREDICTION_TABLE)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .select("MODEL_VERSION", F.col("ENTITY_MAP")[TIME_COL].alias("ENTITY_TIME"))
        .distinct()
        .collect()
    )
)

# Build the list of (version, week) pairs that need inference
combos_needed = {}
for version in versions_to_run:
    missing_weeks = [
        tv for tv in all_time_values
        if (version, tv) not in existing_combos
    ]
    if missing_weeks:
        combos_needed[version] = missing_weeks
    else:
        print(f"  {version}: all weeks already in {PREDICTION_TABLE}")

print(f"\nVersions from active tags (deduplicated): {versions_to_run}")
print(f"Version-week combos needing inference: "
      f"{sum(len(ws) for ws in combos_needed.values())}")

# %% [markdown]
# ### 3B. Run partitioned inference by week
#
# For each candidate version, executes partitioned SQL inference by STATS_NTILE_GROUP
# over each missing week. Each batch is loaded as a temporary view (BATCH_PAGE) and
# predictions are inserted directly into DA_PREDICTIONS.

# %%
input_col_refs = ",\n        ".join(f"{c} => t.{c}" for c in PREDICT_INPUT_COLS)
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

# -- SQL fragments -----------------------------------------------------
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

for version, missing_weeks in combos_needed.items():
    print(f"\nRunning inference for version: {version}")
    start_time = time.time()

    insert_batch_sql = f"""
    INSERT INTO {PREDICTION_TABLE}
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
        p.{PREDICTION_COL.lower()} AS PREDICTION,
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

    print(f"  {len(missing_weeks)} {TIME_COL} value(s) to process")

    for i, tv in enumerate(missing_weeks):
        batch_df = features_df.filter(F.col(TIME_COL) == tv)
        if INFERENCE_SAMPLE_FRACTION is not None and 0 < INFERENCE_SAMPLE_FRACTION < 1:
            batch_df = batch_df.sample(frac=INFERENCE_SAMPLE_FRACTION)

        batch_df.create_or_replace_temp_view("BATCH_PAGE")
        print(f"    {TIME_COL}={tv} ({i+1}/{len(missing_weeks)})")

        session.sql(insert_batch_sql).collect()

    elapsed = time.time() - start_time
    count = (
        session.table(PREDICTION_TABLE)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(F.col("MODEL_VERSION") == version)
        .count()
    )
    print(f"  {count:,} predictions saved for version {version} in {elapsed:.1f}s")

if not combos_needed:
    print("No new version-week combos need inference.")

# %%
print(insert_batch_sql)

# %% [markdown]
# ### 3C. Summary
#
# Final summary: shows the prediction count per version and the total
# stored in DA_PREDICTIONS for this model.

# %%
for version in versions_to_run:
    count = (
        session.table(PREDICTION_TABLE)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(F.col("MODEL_VERSION") == version)
        .count()
    )
    print(f"  {version}: {count:,} predictions")

total = (
    session.table(PREDICTION_TABLE)
    .filter(F.col("MODEL_NAME") == MODEL_NAME)
    .count()
)
print(f"\nTotal predictions in {PREDICTION_TABLE}: {total:,}")

# %% [markdown]
# ## 4. Post-inference setup
#
# Creates final views and clones ground truth data.

# %%
session.sql(f"""
CREATE OR REPLACE TRANSIENT TABLE {DA_PREDICTIONS_VW} AS
SELECT
    vw.*,
    mp.stats_ntile_group,
    mp.cust_category
FROM {PREDICTION_TABLE} AS vw
LEFT JOIN {INFERENCE_CUST_CATEGORY_LOOKUP} AS mp
ON vw.entity_map:customer_id = mp.customer_id
    AND vw.entity_map:brand_pres_ret = mp.brand_pres_ret
    AND vw.entity_map:week = mp.week
""").collect()

session.sql(f"""
CREATE OR REPLACE TRANSIENT TABLE {GROUND_TRUTH_DATASET_STRUCTURED}
CLONE {DATABASE}.{SRC_STORAGE_SCHEMA}.{GROUND_TRUTH_DATASET_STRUCTURED}
""").collect()

session.sql(f"""
CREATE OR REPLACE VIEW {ACTUALS_TABLE_VW} AS
SELECT *
FROM {GROUND_TRUTH_DATASET_STRUCTURED} AS td
WHERE week < 202548
""").collect()

print(f"Post-inference setup complete: {DA_PREDICTIONS_VW}, {GROUND_TRUTH_DATASET_STRUCTURED}, {ACTUALS_TABLE_VW}")

# %% [markdown]
# ## 5. (Optional) Sample Inference via Registry (Python/pandas)
#
# Useful for debugging, quick validation or inspecting model output
# directly in Python, using the versions identified in Section 3A.

# %%
print("\n🔍 OPTIONAL: SAMPLE PREDICTION VIA REGISTRY")

try:
    # Take a small sample of input to avoid loading everything into memory
    sample_sp = features_df.limit(10)
    sample_pdf = sample_sp.to_pandas()

    for version in versions_to_run:
        print(f"\n--- Loading version: {version} ---")
        model_v = model_ref.version(version)
        # Use force=True to skip package version validation if local environment differs
        local_model = model_v.load(force=True)

        # Execute local prediction using CustomModel
        sample_pred_pdf = local_model.predict(sample_pdf)

        print(f"✅ Sample prediction for {version} completed")
        print(sample_pred_pdf.head())

except Exception as e:
    print(f"ℹ️  Skipping optional local prediction via Registry: {e}")
    print("\n⚠️  Local prediction requires installing the model dependencies.")
    print("   To find them in Snowflake:")
    print("   1. Go to the Model Registry in Snowsight.")
    print("   2. Open the model:")
    print("      BD_AA_DEV.SC_MODELS_BMX.UNI_BOX_REGRESSION_PARTITIONED")
    print("   3. Select the version you are trying to load.")
    print("   4. Download the model artifacts.")
    print("   5. Open the file:")
    print("      model/env/conda.yml")
    print("   6. Install the listed packages locally (pip or conda).")
    print("\n   Note: The main SQL-based inference above works without these local dependencies.")


