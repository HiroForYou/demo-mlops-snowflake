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

# Set context
session.sql("USE DATABASE BD_AA_DEV").collect()
session.sql("USE SCHEMA SC_STORAGE_BMX_PS").collect()

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
    train_df = session.table("BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_STRUCTURED")
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
columns_lower = [col.lower() for col in columns]
if "uni_box_week" in columns_lower:
    # Find the actual column name (preserving case)
    target_col = columns[columns_lower.index("uni_box_week")]
    print(f"\n✅ Target variable 'uni_box_week' found (as '{target_col}')")
else:
    print(f"\n❌ Target variable 'uni_box_week' NOT found!")
    print(f"   Available columns: {', '.join(columns)}")
    raise ValueError("Target variable 'uni_box_week' is required")

# Store target column name for later use
TARGET_COLUMN = target_col

# %% [markdown]
# ## 2. Validate Inference Dataset

# %%
print("\n" + "=" * 80)
print("📊 VALIDATING INFERENCE DATASET")
print("=" * 80)

try:
    inference_df = session.table(
        "BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_DATASET_STRUCTURED"
    )
    inference_rows = inference_df.count()
    print(f"\n✅ Table exists: INFERENCE_DATASET_STRUCTURED")
    print(f"   Total rows: {inference_rows:,}")
except Exception as e:
    print(f"\n❌ Error accessing table: {str(e)}")
    raise

# Verify target is NOT in inference
inference_columns = inference_df.columns
inference_columns_lower = [col.lower() for col in inference_columns]
if "uni_box_week" in inference_columns_lower:
    print(f"\n⚠️  WARNING: Target variable 'uni_box_week' found in inference dataset")
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
    """
    SELECT
        COUNT(*) AS TOTAL_ROWS,
        SUM(CASE WHEN uni_box_week IS NULL THEN 1 ELSE 0 END) AS NULL_TARGET,
        SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) AS NULL_CUSTOMER_ID,
        SUM(CASE WHEN week IS NULL THEN 1 ELSE 0 END) AS NULL_WEEK
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_STRUCTURED
"""
)

print("\n📊 NULL Values in Training Data:")
null_check_train.show()

# Check for NULLs in key features
feature_null_check = session.sql(
    """
    SELECT
        SUM(CASE WHEN sum_past_12_weeks IS NULL THEN 1 ELSE 0 END) AS NULL_sum_past_12_weeks,
        SUM(CASE WHEN avg_past_12_weeks IS NULL THEN 1 ELSE 0 END) AS NULL_avg_past_12_weeks,
        SUM(CASE WHEN week_of_year IS NULL THEN 1 ELSE 0 END) AS NULL_week_of_year,
        SUM(CASE WHEN stats_ntile_group IS NULL THEN 1 ELSE 0 END) AS NULL_stats_ntile_group
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_STRUCTURED
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
    """
    SELECT
        COUNT(*) AS TOTAL_RECORDS,
        COUNT(DISTINCT uni_box_week) AS UNIQUE_VALUES,
        MIN(uni_box_week) AS MIN_VALUE,
        MAX(uni_box_week) AS MAX_VALUE,
        AVG(uni_box_week) AS MEAN_VALUE,
        STDDEV(uni_box_week) AS STDDEV_VALUE,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY uni_box_week) AS Q1,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY uni_box_week) AS MEDIAN,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY uni_box_week) AS Q3
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_STRUCTURED
    WHERE uni_box_week IS NOT NULL
"""
)

print("\n📊 Target Variable (uni_box_week) Statistics:")
target_stats.show()

# Check for outliers (values beyond 3 standard deviations)
outlier_check = session.sql(
    """
    WITH stats AS (
        SELECT
            AVG(uni_box_week) AS mean_val,
            STDDEV(uni_box_week) AS stddev_val
        FROM BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_STRUCTURED
        WHERE uni_box_week IS NOT NULL
    )
    SELECT
        COUNT(*) AS OUTLIER_COUNT,
        MIN(uni_box_week) AS MIN_OUTLIER,
        MAX(uni_box_week) AS MAX_OUTLIER
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_STRUCTURED, stats
    WHERE uni_box_week IS NOT NULL
        AND (uni_box_week < mean_val - 3 * stddev_val 
             OR uni_box_week > mean_val + 3 * stddev_val)
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

# Define excluded columns
excluded_cols = [
    "customer_id",
    "brand_pres_ret",
    "week",
    "group",
    "stats_group",
    "percentile_group",
    "stats_ntile_group",
]

# Get feature columns from training (exclude target + excluded)
train_feature_cols = [
    col for col in columns if col not in excluded_cols and col != TARGET_COLUMN
]

# Get feature columns from inference (exclude excluded)
inference_feature_cols = [col for col in inference_columns if col not in excluded_cols]

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

cleaned_train_sql = """
CREATE OR REPLACE TABLE BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED AS
SELECT *
FROM BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_STRUCTURED
WHERE uni_box_week IS NOT NULL
    AND customer_id IS NOT NULL
    AND week IS NOT NULL
    -- Remove extreme outliers (optional - adjust threshold as needed)
    AND uni_box_week >= 0
    AND uni_box_week <= (
        SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY uni_box_week)
        FROM BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_STRUCTURED
        WHERE uni_box_week IS NOT NULL
    )
"""

session.sql(cleaned_train_sql).collect()

cleaned_train_count = session.table(
    "BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED"
).count()
print(f"✅ Cleaned training table created: {cleaned_train_count:,} rows")

# Create cleaned inference table
print("\n📝 Creating cleaned inference table...")

cleaned_inference_sql = """
CREATE OR REPLACE TABLE BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_DATASET_CLEANED AS
SELECT *
FROM BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_DATASET_STRUCTURED
WHERE customer_id IS NOT NULL
    AND week IS NOT NULL
"""

session.sql(cleaned_inference_sql).collect()

cleaned_inference_count = session.table(
    "BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_DATASET_CLEANED"
).count()
print(f"✅ Cleaned inference table created: {cleaned_inference_count:,} rows")

# %% [markdown]
# ## 7. Validate stats_ntile_group Segmentation

# %%
print("\n" + "=" * 80)
print("🔍 VALIDATING stats_ntile_group SEGMENTATION")
print("=" * 80)

# Check if stats_ntile_group exists
if "stats_ntile_group" not in columns_lower:
    print("\n❌ ERROR: Column 'stats_ntile_group' NOT found in training dataset!")
    print("   This column is required for 16-group model training.")
    raise ValueError("stats_ntile_group column is required")

# Find actual column name (preserving case)
stats_ntile_col = columns[columns_lower.index("stats_ntile_group")]

# Get unique groups
groups_df = session.sql(
    f"""
    SELECT 
        {stats_ntile_col} AS GROUP_NAME,
        COUNT(*) AS RECORD_COUNT,
        COUNT(DISTINCT customer_id) AS UNIQUE_CUSTOMERS,
        AVG(uni_box_week) AS AVG_TARGET,
        MIN(uni_box_week) AS MIN_TARGET,
        MAX(uni_box_week) AS MAX_TARGET
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED
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
        FROM BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED
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
    """
    SELECT
        'Training (Original)' AS DATASET,
        COUNT(*) AS TOTAL_ROWS,
        COUNT(DISTINCT customer_id) AS UNIQUE_CUSTOMERS,
        COUNT(DISTINCT week) AS UNIQUE_WEEKS
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_STRUCTURED
    UNION ALL
    SELECT
        'Training (Cleaned)' AS DATASET,
        COUNT(*) AS TOTAL_ROWS,
        COUNT(DISTINCT customer_id) AS UNIQUE_CUSTOMERS,
        COUNT(DISTINCT week) AS UNIQUE_WEEKS
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED
    UNION ALL
    SELECT
        'Inference (Original)' AS DATASET,
        COUNT(*) AS TOTAL_ROWS,
        COUNT(DISTINCT customer_id) AS UNIQUE_CUSTOMERS,
        COUNT(DISTINCT week) AS UNIQUE_WEEKS
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_DATASET_STRUCTURED
    UNION ALL
    SELECT
        'Inference (Cleaned)' AS DATASET,
        COUNT(*) AS TOTAL_ROWS,
        COUNT(DISTINCT customer_id) AS UNIQUE_CUSTOMERS,
        COUNT(DISTINCT week) AS UNIQUE_WEEKS
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_DATASET_CLEANED
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
