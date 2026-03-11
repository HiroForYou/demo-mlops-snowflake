# %% [markdown]
# # Data Validation and Cleaning
#
# Validates and cleans the training and inference datasets before feature
# materialization.  Steps performed:
# 1. Validate table structure and data quality for both datasets.
# 2. Clean data (remove NULLs, cap outliers at the 99th percentile).
# 3. Verify feature column compatibility between training and inference.
# 4. Generate a per-group distribution report for STATS_NTILE_GROUP.

# %% [markdown]
# ## 1. Setup

# %%
from snowflake.snowpark.context import get_active_session

session = get_active_session()

# %% [markdown]
# ### 1A. Constants

# %%
DATABASE = "BD_AA_DEV"
STORAGE_SCHEMA = "SC_STORAGE_BMX_PS"
TRAIN_TABLE_STRUCTURED  = f"{DATABASE}.{STORAGE_SCHEMA}.TRAIN_DATASET_STRUCTURED"
INFERENCE_TABLE_STRUCTURED = f"{DATABASE}.{STORAGE_SCHEMA}.INFERENCE_DATASET_STRUCTURED"
TRAIN_TABLE_CLEANED     = f"{DATABASE}.{STORAGE_SCHEMA}.TRAIN_DATASET_CLEANED"
TRAIN_TABLE_HOLDOUT     = f"{DATABASE}.{STORAGE_SCHEMA}.TRAIN_DATASET_HOLDOUT"
INFERENCE_TABLE_CLEANED = f"{DATABASE}.{STORAGE_SCHEMA}.INFERENCE_DATASET_CLEANED"

TARGET_COLUMN         = "UNI_BOX_WEEK"
STATS_NTILE_GROUP_COL = "STATS_NTILE_GROUP"

HOLDOUT_FRACTION = 0.10  # 10% temporal holdout for baseline drift

# Metadata / identifier columns excluded from the feature set
EXCLUDED_COLS = [
    "CUSTOMER_ID",
    "BRAND_PRES_RET",
    "WEEK",
    STATS_NTILE_GROUP_COL,
    "PROD_KEY",
]

session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()
print(f"Session: {session.get_current_database()}.{session.get_current_schema()}")

# %% [markdown]
# ## 2. Validate Training Dataset

# %%
try:
    train_df = session.table(TRAIN_TABLE_STRUCTURED)
    total_rows = train_df.count()
    print(f"TRAIN_DATASET_STRUCTURED: {total_rows:,} rows")
except Exception as e:
    print(f"Error accessing table: {str(e)}")
    raise

columns = train_df.columns
print(f"Columns ({len(columns)}): {', '.join(columns)}")

if TARGET_COLUMN in columns:
    print(f"Target column found: '{TARGET_COLUMN}'")
else:
    raise ValueError(f"Target variable '{TARGET_COLUMN}' not found in training dataset")

# %% [markdown]
# ## 3. Validate Inference Dataset

# %%
try:
    inference_df = session.table(INFERENCE_TABLE_STRUCTURED)
    inference_rows = inference_df.count()
    print(f"INFERENCE_DATASET_STRUCTURED: {inference_rows:,} rows")
except Exception as e:
    print(f"Error accessing table: {str(e)}")
    raise

inference_columns = inference_df.columns

if TARGET_COLUMN in inference_columns:
    print(f"WARNING: target '{TARGET_COLUMN}' found in inference dataset — expected to be absent")
else:
    print(f"Target column correctly absent from inference dataset")

# %% [markdown]
# ## 4. Data Quality — NULL Values

# %%
print("\nNULL check — training data:")
session.sql(f"""
    SELECT
        COUNT(*) AS TOTAL_ROWS,
        SUM(CASE WHEN {TARGET_COLUMN}   IS NULL THEN 1 ELSE 0 END) AS NULL_TARGET,
        SUM(CASE WHEN CUSTOMER_ID       IS NULL THEN 1 ELSE 0 END) AS NULL_CUSTOMER_ID,
        SUM(CASE WHEN WEEK              IS NULL THEN 1 ELSE 0 END) AS NULL_WEEK,
        SUM(CASE WHEN STATS_NTILE_GROUP IS NULL THEN 1 ELSE 0 END) AS NULL_STATS_NTILE_GROUP
    FROM {TRAIN_TABLE_STRUCTURED}
""").show()

# %% [markdown]
# ## 5. Target Variable Distribution

# %%
print("Target variable statistics:")
session.sql(f"""
    SELECT
        COUNT(*)    AS TOTAL_RECORDS,
        MIN({TARGET_COLUMN})  AS MIN_VALUE,
        MAX({TARGET_COLUMN})  AS MAX_VALUE,
        AVG({TARGET_COLUMN})  AS MEAN_VALUE,
        STDDEV({TARGET_COLUMN}) AS STDDEV_VALUE,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {TARGET_COLUMN}) AS Q1,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {TARGET_COLUMN}) AS MEDIAN,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {TARGET_COLUMN}) AS Q3
    FROM {TRAIN_TABLE_STRUCTURED}
    WHERE {TARGET_COLUMN} IS NOT NULL
""").show()

print("Outliers (> 3 std dev):")
session.sql(f"""
    WITH stats AS (
        SELECT AVG({TARGET_COLUMN}) AS mean_val, STDDEV({TARGET_COLUMN}) AS stddev_val
        FROM {TRAIN_TABLE_STRUCTURED}
        WHERE {TARGET_COLUMN} IS NOT NULL
    )
    SELECT COUNT(*) AS OUTLIER_COUNT,
           MIN({TARGET_COLUMN}) AS MIN_OUTLIER,
           MAX({TARGET_COLUMN}) AS MAX_OUTLIER
    FROM {TRAIN_TABLE_STRUCTURED}, stats
    WHERE {TARGET_COLUMN} IS NOT NULL
      AND ({TARGET_COLUMN} < mean_val - 3 * stddev_val
           OR {TARGET_COLUMN} > mean_val + 3 * stddev_val)
""").show()

# %% [markdown]
# ## 6. Feature Compatibility Check

# %%
excluded_cols_set = set(EXCLUDED_COLS)
train_feature_cols = [
    col for col in columns
    if col not in excluded_cols_set and col != TARGET_COLUMN
]
inference_feature_cols = [
    col for col in inference_columns
    if col not in excluded_cols_set
]

missing_in_inference = set(train_feature_cols) - set(inference_feature_cols)
missing_in_train     = set(inference_feature_cols) - set(train_feature_cols)

if missing_in_inference:
    print(f"Features in training but NOT in inference: {sorted(missing_in_inference)}")
if missing_in_train:
    print(f"Features in inference but NOT in training: {sorted(missing_in_train)}")
if not missing_in_inference and not missing_in_train:
    print(f"All features match ({len(train_feature_cols)} features)")

# %% [markdown]
# ## 7. Create Cleaned Tables

# %%
# Split the cleaned data temporally by group: newest X% becomes the holdout set
# We use a TEMPORARY TABLE of thresholds to avoid Data Skew when ordering 
# millions of rows in a few partitions. 
# 1. We count records per group and per week (fast aggregation).
# 2. We calculate the cumulative sum of records to find the 90% threshold week.
temp_thresholds_table = f"{TRAIN_TABLE_CLEANED}_TEMP_THRESHOLDS"

session.sql(f"""
    CREATE TEMPORARY TABLE {temp_thresholds_table} AS
    WITH base_filtered AS (
        SELECT {STATS_NTILE_GROUP_COL}, WEEK, COUNT(*) as weekly_rows
        FROM {TRAIN_TABLE_STRUCTURED}
        WHERE {TARGET_COLUMN} IS NOT NULL
          AND CUSTOMER_ID IS NOT NULL
          AND WEEK IS NOT NULL
          AND {TARGET_COLUMN} >= 0
          AND {TARGET_COLUMN} <= (
              SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY {TARGET_COLUMN})
              FROM {TRAIN_TABLE_STRUCTURED}
              WHERE {TARGET_COLUMN} IS NOT NULL
          )
        GROUP BY {STATS_NTILE_GROUP_COL}, WEEK
    ),
    cumulative_counts AS (
        SELECT {STATS_NTILE_GROUP_COL},
               WEEK,
               weekly_rows,
               SUM(weekly_rows) OVER (PARTITION BY {STATS_NTILE_GROUP_COL} ORDER BY WEEK ASC) as running_total,
               SUM(weekly_rows) OVER (PARTITION BY {STATS_NTILE_GROUP_COL}) as total_group_rows
        FROM base_filtered
    ),
    percentiles AS (
        SELECT {STATS_NTILE_GROUP_COL},
               WEEK,
               running_total / total_group_rows as time_percentile
        FROM cumulative_counts
    ),
    thresholds AS (
        SELECT {STATS_NTILE_GROUP_COL}, MIN(WEEK) as cutoff_week
        FROM percentiles
        WHERE time_percentile >= (1.0 - {HOLDOUT_FRACTION})
        GROUP BY {STATS_NTILE_GROUP_COL}
    )
    SELECT * FROM thresholds
""").collect()

# 3. Create the Training Cleaned Table (<= cutoff_week)
session.sql(f"""
    CREATE OR REPLACE TABLE {TRAIN_TABLE_CLEANED} AS
    SELECT t.*
    FROM {TRAIN_TABLE_STRUCTURED} t
    JOIN {temp_thresholds_table} th
      ON t.{STATS_NTILE_GROUP_COL} = th.{STATS_NTILE_GROUP_COL}
    WHERE t.{TARGET_COLUMN} IS NOT NULL
      AND t.CUSTOMER_ID IS NOT NULL
      AND t.WEEK IS NOT NULL
      AND t.{TARGET_COLUMN} >= 0
      AND t.{TARGET_COLUMN} <= (
          SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY t2.{TARGET_COLUMN})
          FROM {TRAIN_TABLE_STRUCTURED} t2
          WHERE t2.{TARGET_COLUMN} IS NOT NULL
      )
      AND t.WEEK <= th.cutoff_week
""").collect()

# 4. Create the Holdout Dataset (> cutoff_week)
session.sql(f"""
    CREATE OR REPLACE TABLE {TRAIN_TABLE_HOLDOUT} AS
    SELECT t.*
    FROM {TRAIN_TABLE_STRUCTURED} t
    JOIN {temp_thresholds_table} th
      ON t.{STATS_NTILE_GROUP_COL} = th.{STATS_NTILE_GROUP_COL}
    WHERE t.{TARGET_COLUMN} IS NOT NULL
      AND t.CUSTOMER_ID IS NOT NULL
      AND t.WEEK IS NOT NULL
      AND t.{TARGET_COLUMN} >= 0
      AND t.{TARGET_COLUMN} <= (
          SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY t3.{TARGET_COLUMN})
          FROM {TRAIN_TABLE_STRUCTURED} t3
          WHERE t3.{TARGET_COLUMN} IS NOT NULL
      )
      AND t.WEEK > th.cutoff_week
""").collect()

cleaned_train_count   = session.table(TRAIN_TABLE_CLEANED).count()
cleaned_holdout_count = session.table(TRAIN_TABLE_HOLDOUT).count()
print(f"TRAIN_DATASET_CLEANED: {cleaned_train_count:,} rows (Train {1.0 - HOLDOUT_FRACTION:.0%})")
print(f"TRAIN_DATASET_HOLDOUT: {cleaned_holdout_count:,} rows (Holdout {HOLDOUT_FRACTION:.0%})")

session.sql(f"""
    CREATE OR REPLACE TABLE {INFERENCE_TABLE_CLEANED} AS
    SELECT *
    FROM {INFERENCE_TABLE_STRUCTURED}
    WHERE CUSTOMER_ID IS NOT NULL
      AND WEEK IS NOT NULL
""").collect()
cleaned_inference_count = session.table(INFERENCE_TABLE_CLEANED).count()
print(f"INFERENCE_DATASET_CLEANED: {cleaned_inference_count:,} rows")

# %% [markdown]
# ## 8. Validate STATS_NTILE_GROUP Segmentation

# %%
if STATS_NTILE_GROUP_COL not in columns:
    raise ValueError(f"Column '{STATS_NTILE_GROUP_COL}' not found — required for 16-group training")

print("Group distribution:")
session.sql(f"""
    SELECT
        {STATS_NTILE_GROUP_COL} AS GROUP_NAME,
        COUNT(*)                   AS RECORD_COUNT,
        COUNT(DISTINCT CUSTOMER_ID) AS UNIQUE_CUSTOMERS,
        AVG({TARGET_COLUMN})       AS AVG_TARGET
    FROM {TRAIN_TABLE_CLEANED}
    WHERE {STATS_NTILE_GROUP_COL} IS NOT NULL
    GROUP BY {STATS_NTILE_GROUP_COL}
    ORDER BY {STATS_NTILE_GROUP_COL}
""").show()

group_count = session.sql(f"""
    SELECT COUNT(DISTINCT {STATS_NTILE_GROUP_COL}) AS CNT
    FROM {TRAIN_TABLE_CLEANED}
    WHERE {STATS_NTILE_GROUP_COL} IS NOT NULL
""").collect()[0]["CNT"]

if group_count != 16:
    print(f"WARNING: Expected 16 groups, found {group_count}")
else:
    print(f"Validation passed: {group_count} groups found")

min_records_result = session.sql(f"""
    SELECT MIN(RECORD_COUNT) AS MIN_RECORDS, MAX(RECORD_COUNT) AS MAX_RECORDS
    FROM (
        SELECT {STATS_NTILE_GROUP_COL}, COUNT(*) AS RECORD_COUNT
        FROM {TRAIN_TABLE_CLEANED}
        WHERE {STATS_NTILE_GROUP_COL} IS NOT NULL
        GROUP BY {STATS_NTILE_GROUP_COL}
    )
""").collect()[0]
min_records = min_records_result["MIN_RECORDS"]

if min_records < 100:
    print(f"WARNING: Some groups have fewer than 100 records (min: {min_records})")
else:
    print(f"All groups have sufficient data (min: {min_records} records)")

# %% [markdown]
# ## 9. Summary

# %%
print("Dataset comparison:")
session.sql(f"""
    SELECT 'Training (Original)'  AS DATASET, COUNT(*) AS TOTAL_ROWS,
           COUNT(DISTINCT CUSTOMER_ID) AS UNIQUE_CUSTOMERS, COUNT(DISTINCT WEEK) AS UNIQUE_WEEKS
    FROM {TRAIN_TABLE_STRUCTURED}
    UNION ALL
    SELECT 'Training (Cleaned)',   COUNT(*), COUNT(DISTINCT CUSTOMER_ID), COUNT(DISTINCT WEEK)
    FROM {TRAIN_TABLE_CLEANED}
    UNION ALL
    SELECT 'Inference (Original)', COUNT(*), COUNT(DISTINCT CUSTOMER_ID), COUNT(DISTINCT WEEK)
    FROM {INFERENCE_TABLE_STRUCTURED}
    UNION ALL
    SELECT 'Inference (Cleaned)',  COUNT(*), COUNT(DISTINCT CUSTOMER_ID), COUNT(DISTINCT WEEK)
    FROM {INFERENCE_TABLE_CLEANED}
""").show()

print("Data validation and cleaning complete.")
print(f"   Training rows (cleaned {(1-HOLDOUT_FRACTION)*100:.0f}%): {cleaned_train_count:,}")
print(f"   Holdout rows (cleaned {HOLDOUT_FRACTION*100:.0f}%):  {cleaned_holdout_count:,}")
print(f"   Inference rows (cleaned):    {cleaned_inference_count:,}")
print(f"   STATS_NTILE_GROUP groups: {group_count}")
print(f"   Minimum records per group (Train): {min_records}")
print("\nNext: 02_feature_store_setup.py")
