# %% [markdown]
# # Data Validation and Cleaning
#
# Validates and cleans the training and inference datasets before feature
# materialization.  Steps performed:
# 1. Validate table structure and data quality for both datasets.
# 2. Clean data (remove NULLs, cap outliers at the 99th percentile).
#    Note: "cap" is implemented as a filter that DROPS rows whose target
#    exceeds the computed 99th-percentile threshold (global across the
#    training dataset, not per group).
#    The outlier recoding/filter is OPTIONAL and controlled by
#    `APPLY_OUTLIER_FILTER_P99` (default: False).
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
FEATURES_SCHEMA = "SC_FEATURES_BMX"

# Feature store name (shared by entity and frequency, not tied to model)
FEATURE_STORE_NAME = "FEAT_CUSTBPR_WEEKLY"

# Source tables (structured raw data)
TRAIN_TABLE_STRUCTURED  = f"{DATABASE}.{STORAGE_SCHEMA}.TRAIN_DATASET_STRUCTURED"
INFERENCE_TABLE_STRUCTURED = f"{DATABASE}.{STORAGE_SCHEMA}.INFERENCE_DATASET_STRUCTURED"

# Target feature tables (cleaned and split)
TRAIN_TABLE_CLEANED     = f"{DATABASE}.{FEATURES_SCHEMA}.{FEATURE_STORE_NAME}__TRAIN"
TRAIN_TABLE_HOLDOUT     = f"{DATABASE}.{FEATURES_SCHEMA}.{FEATURE_STORE_NAME}__HOLDOUT"
INFERENCE_TABLE_CLEANED = f"{DATABASE}.{FEATURES_SCHEMA}.{FEATURE_STORE_NAME}__INF"

TARGET_COLUMN         = "UNI_BOX_WEEK"
STATS_NTILE_GROUP_COL = "STATS_NTILE_GROUP"

HOLDOUT_FRACTION = 0.10  # 10% temporal holdout for baseline drift

# Optional label outlier cleaning (P99).
# When False, the temporal split is computed without dropping label outliers.
APPLY_OUTLIER_FILTER_P99 = False

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
#
# Ensures the structured training table is reachable, reports row/column
# information, and verifies that the expected target column exists.

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
#
# Confirms that the structured inference table is reachable and that the
# target column is absent (as expected for inference-only datasets).

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
#
# Computes null counts for the target and key identifier columns in the
# training dataset to quantify data quality issues before cleaning.

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
#
# Summarizes the target distribution (min/max/mean/std and percentiles) and
# reports potential outliers as a sanity check prior to label cleaning.

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
# ## Outlier Handling — P99 Threshold (Justification)
#
# The pipeline uses a robust label cleaning strategy based on a P99 target threshold.
# When `APPLY_OUTLIER_FILTER_P99=True`, the P99 threshold is computed after the
# temporal `cutoff_week` is known and is estimated ONLY from the TRAIN window
# (`WEEK <= cutoff_week`). This avoids using holdout label information to estimate
# the cleaning threshold.


# %%
# The actual P99 computation is deferred until after `temp_thresholds_table`
# is created (so we know the TRAIN window precisely). This makes the decision
# auditable and prevents label leakage into the threshold estimation.
p99_threshold = None
if APPLY_OUTLIER_FILTER_P99:
    print(
        "\nOutlier filter (P99): ENABLED. P99 threshold computation will be deferred "
        "until after the temporal cutoff_week is computed, and will be estimated ONLY "
        "from the TRAIN window (WEEK <= cutoff_week)."
    )
else:
    print("\nOutlier filter (P99): DISABLED. No P99 label filtering will be applied.")

# %% [markdown]
# ## 6. Feature Compatibility Check
#
# Verifies that feature columns used for training match those available for
# inference by comparing column sets after excluding identifiers/metadata.

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
# 0. Optional: label outlier filtering (P99) is applied AFTER cutoff_week is computed.
#    If enabled, P99 is estimated ONLY from the TRAIN window (WEEK <= cutoff_week).
# 1. We count records per group and per week (fast aggregation).
# 2. We calculate the cumulative sum of records to find the 90% threshold week.
temp_thresholds_table = f"{TRAIN_TABLE_CLEANED}_TEMP_THRESHOLDS"

# Build optional outlier filter snippets (placeholders overwritten after P99 is computed).
p99_threshold_sql = str(p99_threshold) if p99_threshold is not None else None
outlier_filter_base = (
    f"AND {TARGET_COLUMN} <= {p99_threshold_sql}" if APPLY_OUTLIER_FILTER_P99 else ""
)
outlier_filter_t = (
    f"AND t.{TARGET_COLUMN} <= {p99_threshold_sql}" if APPLY_OUTLIER_FILTER_P99 else ""
)

session.sql(f"""
    CREATE OR REPLACE TEMPORARY TABLE {temp_thresholds_table} AS
    WITH base_filtered AS (
        SELECT {STATS_NTILE_GROUP_COL}, WEEK, COUNT(*) as weekly_rows
        FROM {TRAIN_TABLE_STRUCTURED}
        WHERE {TARGET_COLUMN} IS NOT NULL
          AND CUSTOMER_ID IS NOT NULL
          AND WEEK IS NOT NULL
          AND {STATS_NTILE_GROUP_COL} IS NOT NULL
          AND {TARGET_COLUMN} >= 0
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

# 2b. Compute P99 (optional) using ONLY the TRAIN window (no holdout leakage)
if APPLY_OUTLIER_FILTER_P99:
    p99_threshold = session.sql(f"""
        SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY t.{TARGET_COLUMN}) AS P99
        FROM {TRAIN_TABLE_STRUCTURED} t
        JOIN {temp_thresholds_table} th
          ON t.{STATS_NTILE_GROUP_COL} = th.{STATS_NTILE_GROUP_COL}
        WHERE t.{TARGET_COLUMN} IS NOT NULL
          AND t.CUSTOMER_ID IS NOT NULL
          AND t.WEEK IS NOT NULL
          AND t.{STATS_NTILE_GROUP_COL} IS NOT NULL
          AND t.{TARGET_COLUMN} >= 0
          AND t.WEEK <= th.cutoff_week
    """).collect()[0]["P99"]

    print(f"\nOutlier filter (P99 computed from TRAIN window): {TARGET_COLUMN} <= {p99_threshold}")

    # Update SQL snippets used to filter TRAIN/HOLDOUT rows.
    outlier_filter_base = f"AND {TARGET_COLUMN} <= {p99_threshold}"
    outlier_filter_t = f"AND t.{TARGET_COLUMN} <= {p99_threshold}"

    outlier_counts = session.sql(f"""
        SELECT
            COUNT(*) AS candidate_rows,
            SUM(CASE WHEN {TARGET_COLUMN} > {p99_threshold} THEN 1 ELSE 0 END) AS outlier_rows_removed
        FROM {TRAIN_TABLE_STRUCTURED}
        WHERE {TARGET_COLUMN} IS NOT NULL
          AND CUSTOMER_ID IS NOT NULL
          AND WEEK IS NOT NULL
          AND {STATS_NTILE_GROUP_COL} IS NOT NULL
          AND {TARGET_COLUMN} >= 0
    """).collect()[0]

    print(
        "Rows removed as outliers (target > P99): "
        f"{outlier_counts['OUTLIER_ROWS_REMOVED']:,} / {outlier_counts['CANDIDATE_ROWS']:,}"
    )

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
      {outlier_filter_t}
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
      {outlier_filter_t}
      AND t.WEEK > th.cutoff_week
""").collect()

cleaned_train_count   = session.table(TRAIN_TABLE_CLEANED).count()
cleaned_holdout_count = session.table(TRAIN_TABLE_HOLDOUT).count()
print(f"TRAIN_DATASET_CLEANED: {cleaned_train_count:,} rows (Train {1.0 - HOLDOUT_FRACTION:.0%})")
print(f"TRAIN_DATASET_HOLDOUT: {cleaned_holdout_count:,} rows (Holdout {HOLDOUT_FRACTION:.0%})")

# Consistency check: temporal split should not drop/duplicate rows.
# We compare against the "candidate" set used by `base_filtered` (i.e., original
# TRAIN_TABLE_STRUCTURED after applying label cleaning + NULL checks), so
# TRAIN + HOLDOUT should sum to that candidate total.
cleaned_total_count = cleaned_train_count + cleaned_holdout_count
candidate_total_count = session.sql(f"""
    SELECT COUNT(*) AS candidate_rows
    FROM {TRAIN_TABLE_STRUCTURED}
    WHERE {TARGET_COLUMN} IS NOT NULL
      AND CUSTOMER_ID IS NOT NULL
      AND WEEK IS NOT NULL
      AND {STATS_NTILE_GROUP_COL} IS NOT NULL
      AND {TARGET_COLUMN} >= 0
      {outlier_filter_base}
""").collect()[0]["CANDIDATE_ROWS"]

raw_training_total_count = total_rows
split_consistency_diff = cleaned_total_count - candidate_total_count
split_consistency_ok = (split_consistency_diff == 0)

cleaned_vs_raw_diff = cleaned_total_count - raw_training_total_count
cleaned_vs_raw_ok = (cleaned_vs_raw_diff == 0)

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
#
# Audits the per-group segmentation to ensure the expected number of groups
# exists and that each group has sufficient training records.

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

print("\nTemporal split consistency (audit):")
print(f"   Candidate rows: {candidate_total_count:,}")
print(f"   Cleaned train+holdout rows: {cleaned_total_count:,}")
if split_consistency_ok:
    print("   OK: candidate_total_count == cleaned_total_count")
else:
    print(
        f"   WARNING: candidate vs cleaned mismatch. "
        f"(cleaned_total - candidate_total) = {split_consistency_diff:,}"
    )

print(f"   Original training structured rows: {raw_training_total_count:,}")
if cleaned_vs_raw_ok:
    print("   OK: cleaned_total_count == raw_training_total_count")
else:
    print(
        "   NOTE: Cleaned train+holdout do not sum to original training structured. "
        f"(cleaned_total - raw_training_total_count) = {cleaned_vs_raw_diff:,}. "
        "This is expected when data cleaning removes rows before the temporal split "
        "(NULL target / NULL CUSTOMER_ID / NULL WEEK / target < 0 / target > P99 / "
        "and rows with STATS_NTILE_GROUP IS NULL)."
    )

print(f"\nOutlier P99 filter enabled? {APPLY_OUTLIER_FILTER_P99}")
if APPLY_OUTLIER_FILTER_P99:
    print(f"   P99 threshold used: {p99_threshold}")
else:
    pass

holdout_share_pct = (cleaned_holdout_count / cleaned_total_count * 100.0) if cleaned_total_count else 0.0

print(f"   Actual holdout share (by rows): {holdout_share_pct:.2f}%")

print("\nTemporal cutoff week per group (TRAIN uses WEEK <= cutoff_week):")
session.sql(f"""
    SELECT
        {STATS_NTILE_GROUP_COL} AS GROUP_NAME,
        cutoff_week
    FROM {temp_thresholds_table}
    ORDER BY {STATS_NTILE_GROUP_COL}
""").show()

# %% [markdown]
# ### Audit notes: temporal split + P99 label cleaning
#
# #### 1) Why the split is not exactly 10%
# - This split is temporal and is computed by whole `WEEK` periods per `STATS_NTILE_GROUP`.
# - `TRAIN` uses `WEEK <= cutoff_week` and `HOLDOUT` uses `WEEK > cutoff_week`.
# - `cutoff_week` is chosen as the first week where the cumulative share reaches/exceeds
#   the threshold `(1 - HOLDOUT_FRACTION)`.
# - Because volumes are aggregated by weeks (not per-row), the resulting holdout share can differ
#   slightly from the configured 10% (e.g., 9.23%).
#
# #### 2) Label leakage risk when `APPLY_OUTLIER_FILTER_P99=True`
# - When enabled, the P99 threshold is computed ONLY from the TRAIN window (`WEEK <= cutoff_week`).
# - The holdout period is NOT used to estimate the threshold; therefore, the cleaning rule
#   does not depend on holdout labels.
# - After the threshold is fixed, the same P99 rule is applied consistently to both TRAIN and
#   HOLDOUT rows to cap extreme label values.
