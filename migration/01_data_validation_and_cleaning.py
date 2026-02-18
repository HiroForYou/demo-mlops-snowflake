# %% [markdown]
# # Migration: Data Validation and Cleaning
#
# ## Overview
# This script validates and cleans the training and inference datasets before creating the Feature Store.
#
# ## What We'll Do:
# 1. Validate table structures and data quality
# 2. Clean data (handle NULLs, outliers)
# 3. Verify feature compatibility between train and inference
# 4. Generate data quality reports

# %%
from snowflake.snowpark.context import get_active_session

session = get_active_session()

# Configuration: Database, schemas, and tables
DATABASE = "BD_AA_DEV"
STORAGE_SCHEMA = "SC_STORAGE_BMX_PS"
TRAIN_TABLE_STRUCTURED = f"{DATABASE}.{STORAGE_SCHEMA}.TRAIN_DATASET_STRUCTURED"
INFERENCE_TABLE_STRUCTURED = f"{DATABASE}.{STORAGE_SCHEMA}.INFERENCE_DATASET_STRUCTURED"
TRAIN_TABLE_CLEANED = f"{DATABASE}.{STORAGE_SCHEMA}.TRAIN_DATASET_CLEANED"
INFERENCE_TABLE_CLEANED = f"{DATABASE}.{STORAGE_SCHEMA}.INFERENCE_DATASET_CLEANED"

# Column constants
TARGET_COLUMN = "UNI_BOX_WEEK"
STATS_NTILE_GROUP_COL = "STATS_NTILE_GROUP"

# Excluded columns (metadata columns, not features) - defined once at the beginning
EXCLUDED_COLS = [
    "CUSTOMER_ID",
    "BRAND_PRES_RET",
    "WEEK",
    "GROUP",
    "STATS_GROUP",
    "PERCENTILE_GROUP",
    STATS_NTILE_GROUP_COL,
    "PROD_KEY",
]

# Set context
session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()

print(f"✅ Connected to Snowflake")
print(f"   Database: {session.get_current_database()}")
print(f"   Schema: {session.get_current_schema()}")

# %% [markdown]
# ## 1. Validate Training Dataset

# %%
print("\n" + "=" * 80)
print("📊 VALIDATING TRAINING DATASET")
print("=" * 80)

# Check if table exists
try:
    train_df = session.table(TRAIN_TABLE_STRUCTURED)
    total_rows = train_df.count()
    print(f"\n✅ Table exists: TRAIN_DATASET_STRUCTURED")
    print(f"   Total rows: {total_rows:,}")
except Exception as e:
    print(f"\n❌ Error accessing table: {str(e)}")
    raise

# Get column information
columns = train_df.columns
print(f"\n📋 Columns ({len(columns)}):")
for col in columns:
    print(f"   - {col}")

# Check for target variable (case-insensitive)
columns_upper = [col.upper() for col in columns]
if TARGET_COLUMN in columns_upper:
    # Find the actual column name (preserving case)
    target_col = columns[columns_upper.index(TARGET_COLUMN)]
    print(f"\n✅ Target variable '{TARGET_COLUMN}' found (as '{target_col}')")
else:
    print(f"\n❌ Target variable '{TARGET_COLUMN}' NOT found!")
    print(f"   Available columns: {', '.join(columns)}")
    raise ValueError(f"Target variable '{TARGET_COLUMN}' is required")

# %% [markdown]
# ## 2. Validate Inference Dataset

# %%
print("\n" + "=" * 80)
print("📊 VALIDATING INFERENCE DATASET")
print("=" * 80)

try:
    inference_df = session.table(INFERENCE_TABLE_STRUCTURED)
    inference_rows = inference_df.count()
    print(f"\n✅ Table exists: INFERENCE_DATASET_STRUCTURED")
    print(f"   Total rows: {inference_rows:,}")
except Exception as e:
    print(f"\n❌ Error accessing table: {str(e)}")
    raise

# Verify target is NOT in inference
inference_columns = inference_df.columns
inference_columns_upper = [col.upper() for col in inference_columns]
if TARGET_COLUMN in inference_columns_upper:
    print(f"\n⚠️  WARNING: Target variable '{TARGET_COLUMN}' found in inference dataset")
    print(f"   This is expected - inference should not have target values")
else:
    print(f"\n✅ Target variable correctly absent from inference dataset")

# %% [markdown]
# ## 3. Check Data Quality - NULLs and Missing Values

# %%
print("\n" + "=" * 80)
print("🔍 DATA QUALITY CHECK - NULL VALUES")
print("=" * 80)

# Check NULLs in training data
null_check_train = session.sql(
    f"""
    SELECT
        COUNT(*) AS TOTAL_ROWS,
        SUM(CASE WHEN {TARGET_COLUMN} IS NULL THEN 1 ELSE 0 END) AS NULL_TARGET,
        SUM(CASE WHEN CUSTOMER_ID IS NULL THEN 1 ELSE 0 END) AS NULL_CUSTOMER_ID,
        SUM(CASE WHEN WEEK IS NULL THEN 1 ELSE 0 END) AS NULL_WEEK
    FROM {TRAIN_TABLE_STRUCTURED}
"""
)

print("\n📊 NULL Values in Training Data:")
null_check_train.show()

# Check for NULLs in key features
feature_null_check = session.sql(
    f"""
    SELECT
        SUM(CASE WHEN SUM_PAST_12_WEEKS IS NULL THEN 1 ELSE 0 END) AS NULL_SUM_PAST_12_WEEKS,
        SUM(CASE WHEN AVG_PAST_12_WEEKS IS NULL THEN 1 ELSE 0 END) AS NULL_AVG_PAST_12_WEEKS,
        SUM(CASE WHEN WEEK_OF_YEAR IS NULL THEN 1 ELSE 0 END) AS NULL_WEEK_OF_YEAR,
        SUM(CASE WHEN STATS_NTILE_GROUP IS NULL THEN 1 ELSE 0 END) AS NULL_STATS_NTILE_GROUP
    FROM {TRAIN_TABLE_STRUCTURED}
"""
)

print("\n📊 NULL Values in Key Features:")
feature_null_check.show()

# %% [markdown]
# ## 4. Check Target Variable Distribution

# %%
print("\n" + "=" * 80)
print("📈 TARGET VARIABLE DISTRIBUTION")
print("=" * 80)

target_stats = session.sql(
    f"""
    SELECT
        COUNT(*) AS TOTAL_RECORDS,
        COUNT(DISTINCT {TARGET_COLUMN}) AS UNIQUE_VALUES,
        MIN({TARGET_COLUMN}) AS MIN_VALUE,
        MAX({TARGET_COLUMN}) AS MAX_VALUE,
        AVG({TARGET_COLUMN}) AS MEAN_VALUE,
        STDDEV({TARGET_COLUMN}) AS STDDEV_VALUE,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {TARGET_COLUMN}) AS Q1,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {TARGET_COLUMN}) AS MEDIAN,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {TARGET_COLUMN}) AS Q3
    FROM {TRAIN_TABLE_STRUCTURED}
    WHERE {TARGET_COLUMN} IS NOT NULL
"""
)

print("\n📊 Target Variable (uni_box_week) Statistics:")
target_stats.show()

# Check for outliers (values beyond 3 standard deviations)
outlier_check = session.sql(
    f"""
    WITH stats AS (
        SELECT
            AVG({TARGET_COLUMN}) AS mean_val,
            STDDEV({TARGET_COLUMN}) AS stddev_val
        FROM {TRAIN_TABLE_STRUCTURED}
        WHERE {TARGET_COLUMN} IS NOT NULL
    )
    SELECT
        COUNT(*) AS OUTLIER_COUNT,
        MIN({TARGET_COLUMN}) AS MIN_OUTLIER,
        MAX({TARGET_COLUMN}) AS MAX_OUTLIER
    FROM {TRAIN_TABLE_STRUCTURED}, stats
    WHERE {TARGET_COLUMN} IS NOT NULL
        AND ({TARGET_COLUMN} < mean_val - 3 * stddev_val 
             OR {TARGET_COLUMN} > mean_val + 3 * stddev_val)
"""
)

print("\n📊 Outliers (>3 std dev):")
outlier_check.show()

# %% [markdown]
# ## 5. Verify Feature Compatibility

# %%
print("\n" + "=" * 80)
print("🔗 FEATURE COMPATIBILITY CHECK")
print("=" * 80)

# Get feature columns from training (exclude target + excluded)
# EXCLUDED_COLS is already in UPPER CASE, so we compare case-insensitively
excluded_cols_upper = {col for col in EXCLUDED_COLS}
train_feature_cols = [
    col for col in columns 
    if col.upper() not in excluded_cols_upper and col.upper() != TARGET_COLUMN
]

# Get feature columns from inference (exclude excluded)
inference_feature_cols = [
    col for col in inference_columns 
    if col.upper() not in excluded_cols_upper
]

print(f"\n📋 Training Features ({len(train_feature_cols)}):")
for col in sorted(train_feature_cols):
    print(f"   - {col}")

print(f"\n📋 Inference Features ({len(inference_feature_cols)}):")
for col in sorted(inference_feature_cols):
    print(f"   - {col}")

# Check if features match
missing_in_inference = set(train_feature_cols) - set(inference_feature_cols)
missing_in_train = set(inference_feature_cols) - set(train_feature_cols)

if missing_in_inference:
    print(f"\n⚠️  Features in training but NOT in inference:")
    for col in missing_in_inference:
        print(f"   - {col}")

if missing_in_train:
    print(f"\n⚠️  Features in inference but NOT in training:")
    for col in missing_in_train:
        print(f"   - {col}")

if not missing_in_inference and not missing_in_train:
    print(f"\n✅ All features match between training and inference!")

# %% [markdown]
# ## 6. Create Cleaned Tables

# %%
print("\n" + "=" * 80)
print("🧹 CREATING CLEANED TABLES")
print("=" * 80)

# Create cleaned training table
print("\n📝 Creating cleaned training table...")

cleaned_train_sql = f"""
CREATE OR REPLACE TABLE {TRAIN_TABLE_CLEANED} AS
SELECT *
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
"""

session.sql(cleaned_train_sql).collect()

cleaned_train_count = session.table(TRAIN_TABLE_CLEANED).count()
print(f"✅ Cleaned training table created: {cleaned_train_count:,} rows")

# Create cleaned inference table
print("\n📝 Creating cleaned inference table...")

cleaned_inference_sql = f"""
CREATE OR REPLACE TABLE {INFERENCE_TABLE_CLEANED} AS
SELECT *
FROM {INFERENCE_TABLE_STRUCTURED}
WHERE CUSTOMER_ID IS NOT NULL
    AND WEEK IS NOT NULL
"""

session.sql(cleaned_inference_sql).collect()

cleaned_inference_count = session.table(INFERENCE_TABLE_CLEANED).count()
print(f"✅ Cleaned inference table created: {cleaned_inference_count:,} rows")

# %% [markdown]
# ## 7. Validate stats_ntile_group Segmentation

# %%
print("\n" + "=" * 80)
print("🔍 VALIDATING stats_ntile_group SEGMENTATION")
print("=" * 80)

# Check if STATS_NTILE_GROUP exists
if STATS_NTILE_GROUP_COL not in columns_upper:
    print(f"\n❌ ERROR: Column '{STATS_NTILE_GROUP_COL}' NOT found in training dataset!")
    print("   This column is required for 16-group model training.")
    raise ValueError(f"{STATS_NTILE_GROUP_COL} column is required")

# Find actual column name (preserving case)
stats_ntile_col = columns[columns_upper.index(STATS_NTILE_GROUP_COL)]

# Get unique groups
groups_df = session.sql(
    f"""
    SELECT 
        {stats_ntile_col} AS GROUP_NAME,
        COUNT(*) AS RECORD_COUNT,
        COUNT(DISTINCT CUSTOMER_ID) AS UNIQUE_CUSTOMERS,
        AVG({TARGET_COLUMN}) AS AVG_TARGET,
        MIN({TARGET_COLUMN}) AS MIN_TARGET,
        MAX({TARGET_COLUMN}) AS MAX_TARGET
    FROM {TRAIN_TABLE_CLEANED}
    WHERE {stats_ntile_col} IS NOT NULL
    GROUP BY {stats_ntile_col}
    ORDER BY {stats_ntile_col}
"""
)

print("\n📊 Group Distribution:")
groups_df.show()

# Get group count
group_count = groups_df.count()
print(f"\n📊 Total unique groups: {group_count}")

# Validate we have exactly 16 groups
if group_count != 16:
    print(f"\n⚠️  WARNING: Expected 16 groups, found {group_count}")
    print("   This may affect model training. Please verify segmentation logic.")
else:
    print(f"\n✅ Validation passed: Exactly 16 groups found")

# Check minimum records per group (recommend at least 100)
min_records_check = session.sql(
    f"""
    SELECT 
        MIN(RECORD_COUNT) AS MIN_RECORDS,
        MAX(RECORD_COUNT) AS MAX_RECORDS,
        AVG(RECORD_COUNT) AS AVG_RECORDS
    FROM (
        SELECT 
            {stats_ntile_col},
            COUNT(*) AS RECORD_COUNT
        FROM {TRAIN_TABLE_CLEANED}
        WHERE {stats_ntile_col} IS NOT NULL
        GROUP BY {stats_ntile_col}
    )
"""
)

print("\n📊 Records per Group Statistics:")
min_records_check.show()

min_records_result = min_records_check.collect()[0]
min_records = min_records_result["MIN_RECORDS"]

if min_records < 100:
    print(f"\n⚠️  WARNING: Some groups have less than 100 records (minimum: {min_records})")
    print("   This may affect model training quality.")
else:
    print(f"\n✅ All groups have sufficient data (minimum: {min_records} records)")

# %% [markdown]
# ## 8. Summary Statistics

# %%
print("\n" + "=" * 80)
print("📊 SUMMARY STATISTICS")
print("=" * 80)

summary = session.sql(
    f"""
    SELECT
        'Training (Original)' AS DATASET,
        COUNT(*) AS TOTAL_ROWS,
        COUNT(DISTINCT CUSTOMER_ID) AS UNIQUE_CUSTOMERS,
        COUNT(DISTINCT WEEK) AS UNIQUE_WEEKS
    FROM {TRAIN_TABLE_STRUCTURED}
    UNION ALL
    SELECT
        'Training (Cleaned)' AS DATASET,
        COUNT(*) AS TOTAL_ROWS,
        COUNT(DISTINCT CUSTOMER_ID) AS UNIQUE_CUSTOMERS,
        COUNT(DISTINCT WEEK) AS UNIQUE_WEEKS
    FROM {TRAIN_TABLE_CLEANED}
    UNION ALL
    SELECT
        'Inference (Original)' AS DATASET,
        COUNT(*) AS TOTAL_ROWS,
        COUNT(DISTINCT CUSTOMER_ID) AS UNIQUE_CUSTOMERS,
        COUNT(DISTINCT WEEK) AS UNIQUE_WEEKS
    FROM {INFERENCE_TABLE_STRUCTURED}
    UNION ALL
    SELECT
        'Inference (Cleaned)' AS DATASET,
        COUNT(*) AS TOTAL_ROWS,
        COUNT(DISTINCT CUSTOMER_ID) AS UNIQUE_CUSTOMERS,
        COUNT(DISTINCT WEEK) AS UNIQUE_WEEKS
    FROM {INFERENCE_TABLE_CLEANED}
"""
)

print("\n📊 Dataset Comparison:")
summary.show()

print("\n" + "=" * 80)
print("✅ DATA VALIDATION AND CLEANING COMPLETE!")
print("=" * 80)

print("\n📋 Validation Summary:")
print(f"   ✅ Training data validated: {cleaned_train_count:,} rows")
print(f"   ✅ Inference data validated: {cleaned_inference_count:,} rows")
print(f"   ✅ stats_ntile_group validated: {group_count} groups")
print(f"   ✅ Minimum records per group: {min_records}")

print("\n📋 Next Steps:")
print("   1. Review cleaned tables and group distribution")
print("   2. Run 02_feature_store_setup.py to create Feature Store")
print("   3. Run 03_hyperparameter_search.py to find optimal hyperparameters per group")
