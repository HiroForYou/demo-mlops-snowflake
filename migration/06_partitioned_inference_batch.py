# %% [markdown]
# # Partitioned Inference — UNI_BOX_REGRESSION_PARTITIONED
#
# Inference using **PRODUCTION** alias directly, with optional SAMPLE.

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F
from snowflake.ml.registry import Registry
import time
import math

# %%
# Configuration
INFERENCE_SAMPLE_FRACTION = 1

# Number of rows per batch. Leave as None to disable batching.
BATCH_SIZE = 1_000_000

DATABASE = "BD_AA_DEV"
STORAGE_SCHEMA = "SC_STORAGE_BMX_PS"
MODEL_SCHEMA = "SC_MODELS_BMX"
PARTITIONED_MODEL_NAME = "UNI_BOX_REGRESSION_PARTITIONED"
PREDICTIONS_TABLE = f"{DATABASE}.{STORAGE_SCHEMA}.INFERENCE_PREDICTIONS"

SOURCE_TABLE = f"{DATABASE}.{STORAGE_SCHEMA}.INFERENCE_DATASET_CLEANED"
MODEL_FQN = f"{DATABASE}.{MODEL_SCHEMA}.{PARTITIONED_MODEL_NAME}"

# %%
session = get_active_session()
session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()

print("✅ Connected to Snowflake")
print(f"   Model: {PARTITIONED_MODEL_NAME} (PRODUCTION alias)")

# %% [markdown]
# ## Input sampling

# %%
input_df = session.table(SOURCE_TABLE)

if INFERENCE_SAMPLE_FRACTION is not None and 0 < INFERENCE_SAMPLE_FRACTION < 1:
    input_df = input_df.sample(frac=INFERENCE_SAMPLE_FRACTION)
    print(f"⚠️  Using SAMPLE (Snowpark): {INFERENCE_SAMPLE_FRACTION*100:.2f}%")
else:
    print("✅ Using FULL dataset (Snowpark)")

# %%
# Create temporary view of input
input_df.create_or_replace_temp_view("INFERENCE_INPUT")

# %% [markdown]
# ## Partitioned Inference and Save to Table (single-pass, explicit PRODUCTION alias)

# %%
# Get model version information (used to tag predictions)
try:
    registry = Registry(
        session=session,
        database_name=DATABASE,
        schema_name=MODEL_SCHEMA,
    )
    model_ref = registry.get_model(PARTITIONED_MODEL_NAME)
    model_version = model_ref.version("PRODUCTION")
    # ModelVersion may expose the version name under different attributes depending on library version
    model_version_name = getattr(model_version, "version", getattr(model_version, "version_name", str(model_version)))
    print(f"✅ Using model version: {model_version_name}")
except Exception as e:
    print(f"⚠️  Could not get model version: {str(e)[:200]}")
    model_version_name = "UNKNOWN"

# Create predictions table with SQL (including auto-incremental primary key)
create_predictions_table_sql = f"""
CREATE TABLE IF NOT EXISTS {PREDICTIONS_TABLE} (
    PREDICTION_ID NUMBER(38,0) AUTOINCREMENT PRIMARY KEY,
    CUSTOMER_ID VARCHAR,
    STATS_NTILE_GROUP VARCHAR,
    WEEK VARCHAR,
    BRAND_PRES_RET VARCHAR,
    PROD_KEY VARCHAR,
    PREDICTED_UNI_BOX_WEEK FLOAT,
    MODEL_VERSION VARCHAR,
    PREDICTION_TIMESTAMP TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
)
"""

session.sql(create_predictions_table_sql).collect()
print(f"✅ Predictions table ready: {PREDICTIONS_TABLE} (with auto-incremental PRIMARY KEY)")

# %%
# Truncate predictions table before running inference
session.sql(f"TRUNCATE TABLE {PREDICTIONS_TABLE}").collect()
print(f"🧹 Predictions table truncated: {PREDICTIONS_TABLE}")

# %%
# Run partitioned inference and insert predictions into table, optionally in batches
print("🚀 RUNNING PARTITIONED INFERENCE AND SAVING TO TABLE (MODEL(..., PRODUCTION))")
start_time = time.time()

total_rows = input_df.count()
print(f"📊 Input rows (after sampling if applied): {total_rows:,}")

if BATCH_SIZE is None:
    # Inference without batching (single pass)
    insert_predictions_sql = f"""
    INSERT INTO {PREDICTIONS_TABLE}
    (CUSTOMER_ID, STATS_NTILE_GROUP, WEEK, BRAND_PRES_RET, PROD_KEY, PREDICTED_UNI_BOX_WEEK, MODEL_VERSION)
    SELECT
        p.CUSTOMER_ID,
        p.STATS_NTILE_GROUP,
        p.WEEK,
        p.BRAND_PRES_RET,
        p.PROD_KEY,
        p.predicted_uni_box_week,
        '{model_version_name}' AS MODEL_VERSION
    FROM INFERENCE_INPUT t,
    TABLE(
      MODEL({MODEL_FQN}, PRODUCTION)!PREDICT(
        t.CUSTOMER_ID,
        t.STATS_NTILE_GROUP,
        t.WEEK,
        t.BRAND_PRES_RET,
        t.PROD_KEY,
        t.SUM_PAST_12_WEEKS,
        t.AVG_PAST_12_WEEKS,
        t.MAX_PAST_24_WEEKS,
        t.SUM_PAST_24_WEEKS,
        t.WEEK_OF_YEAR,
        t.AVG_AVG_DAILY_ALL_HOURS,
        t.SUM_P4W,
        t.AVG_PAST_24_WEEKS,
        t.PHARM_SUPER_CONV,
        t.WINES_LIQUOR,
        t.GROCERIES,
        t.MAX_PREV2,
        t.AVG_PREV2,
        t.MAX_PREV3,
        t.AVG_PREV3,
        t.W_M1_TOTAL,
        t.W_M2_TOTAL,
        t.W_M3_TOTAL,
        t.W_M4_TOTAL,
        t.SPEC_FOODS,
        t.NUM_COOLERS,
        t.NUM_DOORS,
        t.MAX_PAST_4_WEEKS,
        t.SUM_PAST_4_WEEKS,
        t.AVG_PAST_4_WEEKS,
        t.MAX_PAST_12_WEEKS
      ) OVER (PARTITION BY t.STATS_NTILE_GROUP)
    ) p
    """

    session.sql(insert_predictions_sql).collect()
else:
    # Inference in batches of BATCH_SIZE rows using ROW_NUMBER over the input view
    num_batches = math.ceil(total_rows / BATCH_SIZE) if total_rows > 0 else 0
    print(f"📦 Running in batches of {BATCH_SIZE:,} rows ({num_batches} batch(es))")

    for batch_idx in range(num_batches):
        batch_start = batch_idx * BATCH_SIZE + 1
        batch_end = min((batch_idx + 1) * BATCH_SIZE, total_rows)
        print(f"   ➜ Batch {batch_idx + 1}/{num_batches}: rows {batch_start:,} - {batch_end:,}")

        insert_predictions_sql_batch = f"""
        INSERT INTO {PREDICTIONS_TABLE}
        (CUSTOMER_ID, STATS_NTILE_GROUP, WEEK, BRAND_PRES_RET, PROD_KEY, PREDICTED_UNI_BOX_WEEK, MODEL_VERSION)
        WITH INPUT_BATCH AS (
            SELECT
                *,
                ROW_NUMBER() OVER (ORDER BY CUSTOMER_ID, WEEK, BRAND_PRES_RET, PROD_KEY) AS RN
            FROM INFERENCE_INPUT
        )
        SELECT
            p.CUSTOMER_ID,
            p.STATS_NTILE_GROUP,
            p.WEEK,
            p.BRAND_PRES_RET,
            p.PROD_KEY,
            p.predicted_uni_box_week,
            '{model_version_name}' AS MODEL_VERSION
        FROM INPUT_BATCH t,
        TABLE(
          MODEL({MODEL_FQN}, PRODUCTION)!PREDICT(
            t.CUSTOMER_ID,
            t.STATS_NTILE_GROUP,
            t.WEEK,
            t.BRAND_PRES_RET,
            t.PROD_KEY,
            t.SUM_PAST_12_WEEKS,
            t.AVG_PAST_12_WEEKS,
            t.MAX_PAST_24_WEEKS,
            t.SUM_PAST_24_WEEKS,
            t.WEEK_OF_YEAR,
            t.AVG_AVG_DAILY_ALL_HOURS,
            t.SUM_P4W,
            t.AVG_PAST_24_WEEKS,
            t.PHARM_SUPER_CONV,
            t.WINES_LIQUOR,
            t.GROCERIES,
            t.MAX_PREV2,
            t.AVG_PREV2,
            t.MAX_PREV3,
            t.AVG_PREV3,
            t.W_M1_TOTAL,
            t.W_M2_TOTAL,
            t.W_M3_TOTAL,
            t.W_M4_TOTAL,
            t.SPEC_FOODS,
            t.NUM_COOLERS,
            t.NUM_DOORS,
            t.MAX_PAST_4_WEEKS,
            t.SUM_PAST_4_WEEKS,
            t.AVG_PAST_4_WEEKS,
            t.MAX_PAST_12_WEEKS
          ) OVER (PARTITION BY t.STATS_NTILE_GROUP)
        ) p
        WHERE t.RN BETWEEN {batch_start} AND {batch_end}
        """

        session.sql(insert_predictions_sql_batch).collect()

elapsed = time.time() - start_time

predictions_df = session.table(PREDICTIONS_TABLE).filter(
    F.col("MODEL_VERSION") == model_version_name
)
predictions_count = predictions_df.count()
print(f"✅ {predictions_count:,} predictions saved to {PREDICTIONS_TABLE} in {elapsed:.2f}s (MODEL(..., PRODUCTION))")
print(f"   Model version: {model_version_name}")
predictions_df.order_by("PREDICTION_ID").show(10)

# %% [markdown]
# ## (Optional) Sample Inference via Registry + PRODUCTION (Python/pandas)
#
# Useful for debugging, quick validation or inspecting model output
# directly in Python, using the same version labeled as PRODUCTION.

# %%
print("\n🔍 OPTIONAL: SAMPLE PREDICTION VIA REGISTRY (PRODUCTION)")

try:
    # Create Registry pointing to the same database/schema of models
    registry = Registry(
        session=session,
        database_name=DATABASE,
        schema_name=MODEL_SCHEMA,
    )

    # Get the model and select the PRODUCTION version
    model_ref = registry.get_model(PARTITIONED_MODEL_NAME)
    model_version = model_ref.version("PRODUCTION")
    # Use force=True to skip package version validation if local environment differs
    local_model = model_version.load(force=True)

    # Take a small sample of input to avoid loading everything into memory
    sample_sp = input_df.limit(100)
    sample_pdf = sample_sp.to_pandas()

    # Execute local prediction using CustomModel (PartitionedUniBoxModel)
    sample_pred_pdf = local_model.predict(sample_pdf)

    print("✅ Sample prediction via Registry (PRODUCTION) completed")
    print(sample_pred_pdf.head())
except Exception as e:
    print(f"ℹ️  Skipping optional local prediction via Registry: {e}")
    print("   Note: This requires local installation of model dependencies (lightgbm, xgboost, etc.)")
    print("   The main SQL-based inference above works without these local dependencies.")
