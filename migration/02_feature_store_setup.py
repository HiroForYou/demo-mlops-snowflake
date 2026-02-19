# %% [markdown]
# # Migration: Feature Store Setup (keeping Feature Store, no FeatureView / no Dynamic Tables)
#
# ## Overview
# This script **keeps Feature Store** (schema + `FeatureStore` + `Entity`) but avoids `FeatureView`
# (which may create Dynamic Tables internally) and instead builds and materializes a feature dataset
# as a **normal table** in Snowflake.
#
# ## What We'll Do:
# 1. Create/ensure destination schema for Feature Store
# 2. Initialize `FeatureStore`
# 3. Register `Entity` (optional)
# 4. Build feature dataset from `TRAIN_DATASET_CLEANED`
# 5. Materialize features into a table (CTAS / overwrite)

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.ml.feature_store import FeatureStore, Entity, CreationMode
from datetime import datetime

session = get_active_session()

# Configuration: Database, schemas, and tables
DATABASE = "BD_AA_DEV"
STORAGE_SCHEMA = "SC_STORAGE_BMX_PS"
FEATURES_SCHEMA = "SC_FEATURES_BMX"
TRAIN_TABLE_CLEANED = f"{DATABASE}.{STORAGE_SCHEMA}.TRAIN_DATASET_CLEANED"
FEATURES_TABLE = f"{DATABASE}.{FEATURES_SCHEMA}.UNI_BOX_FEATURES"
FEATURE_VERSIONS_TABLE = f"{DATABASE}.{FEATURES_SCHEMA}.FEATURE_VERSIONS"
DEFAULT_WAREHOUSE = "WH_AA_DEV_DS_SQL"

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
    TARGET_COLUMN,  # Target variable - not a feature
]

# Metadata columns (for feature counting)
METADATA_COLS = ['CUSTOMER_ID', 'BRAND_PRES_RET', 'PROD_KEY', 'WEEK', 'FEATURE_TIMESTAMP']

# Set context
session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()

print(f"✅ Connected to Snowflake")
print(f"   Database: {session.get_current_database()}")
print(f"   Schema: {session.get_current_schema()}")

# %% [markdown]
# ## 1. Create Feature Store Schema

# %%
print("\n" + "=" * 80)
print("🏪 PREPARING FEATURES SCHEMA (NO FEATURE VIEW)")
print("=" * 80)

session.sql(f"USE SCHEMA {FEATURES_SCHEMA}").collect()
print(f"\n✅ Schema ready: {FEATURES_SCHEMA}")

# Initialize Feature Store (without FeatureView)
fs = FeatureStore(
    session=session,
    database=DATABASE,
    name=FEATURES_SCHEMA,
    default_warehouse=DEFAULT_WAREHOUSE,
    creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
)
print("✅ Feature Store initialized (without FeatureView)")

# %% [markdown]
# ## 2. Define Entity (Optional)

# %%
print("\n" + "=" * 80)
print("👤 DEFINING ENTITIES")
print("=" * 80)

# Define Entity (does not create Dynamic Tables; it's Feature Store metadata)
customer_product_entity = Entity(
    name="CUSTOMER_PRODUCT",
    join_keys=["CUSTOMER_ID", "BRAND_PRES_RET", "PROD_KEY"],
    desc="Customer-Product combination entity for UNI_BOX_WEEK regression",
)

try:
    fs.register_entity(customer_product_entity)
    print("✅ Entity 'CUSTOMER_PRODUCT' registered")
except Exception as e:
    print(f"⚠️  Entity may already exist or not be registrable: {str(e)[:120]}")

# %% [markdown]
# ## 3. Build Feature Dataset from Clean Tables

# %%
print("\n" + "=" * 80)
print("📋 BUILDING FEATURE DATASET")
print("=" * 80)

# Get column names efficiently using DESCRIBE TABLE
print("\n⏳ Getting column names from table schema...")
columns_info = session.sql(f"DESCRIBE TABLE {TRAIN_TABLE_CLEANED}").collect()

all_columns = [row["name"] for row in columns_info]

# Get feature columns (all columns except excluded and target)
# EXCLUDED_COLS is already in UPPER CASE, so we compare case-insensitively
excluded_cols_upper = {col for col in EXCLUDED_COLS}
feature_columns = [
    col for col in all_columns 
    if col.upper() not in excluded_cols_upper
]

print(f"\n📋 Column Analysis:")
print(f"   Total columns: {len(all_columns)}")
print(f"   Excluded columns: {len(EXCLUDED_COLS)}")
print(f"   Feature columns: {len(feature_columns)}")

print(f"\n📋 Excluded columns (not features):")
for col in EXCLUDED_COLS:
    print(f"   - {col}")

# Create feature query
# Dynamically select all feature columns (excluding metadata and target)
feature_cols_str = ",\n        ".join(feature_columns)

feature_df = session.sql(
    f"""
    SELECT
        CUSTOMER_ID,
        BRAND_PRES_RET,
        PROD_KEY,
        WEEK,
        {feature_cols_str},
        CASE 
            WHEN WEEK IS NOT NULL THEN 
                TRY_TO_TIMESTAMP_NTZ(WEEK, 'YYYYWW')
            ELSE CURRENT_TIMESTAMP()
        END AS FEATURE_TIMESTAMP
    FROM {TRAIN_TABLE_CLEANED}
    WHERE CUSTOMER_ID IS NOT NULL
        AND BRAND_PRES_RET IS NOT NULL
"""
)

print("✅ Feature query created")

# Count features
feature_count = feature_df.count()
print(f"   Total feature records: {feature_count:,}")

# Register feature version for traceability
print("\n" + "=" * 80)
print("🧾 REGISTERING FEATURE VERSION METADATA")
print("=" * 80)

feature_version_id = f"FV_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
print(f"   Feature version ID: {feature_version_id}")

# Create feature versions table if it does not exist
session.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {FEATURE_VERSIONS_TABLE} (
        FEATURE_VERSION_ID VARCHAR PRIMARY KEY,
        FEATURE_TABLE_NAME VARCHAR,
        CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
        CREATED_BY VARCHAR,
        ROW_COUNT NUMBER,
        FEATURE_COUNT NUMBER,
        SOURCE_TABLE VARCHAR,
        FEATURE_SNAPSHOT_AT TIMESTAMP_NTZ,
        DESCRIPTION VARCHAR,
        IS_ACTIVE BOOLEAN DEFAULT TRUE
    )
"""
).collect()

# Ensure new columns exist if table was created without them in the past (one ALTER per column)
for col_def in [
    "FEATURE_SNAPSHOT_AT TIMESTAMP_NTZ",
    "DESCRIPTION VARCHAR",
    "IS_ACTIVE BOOLEAN DEFAULT TRUE",
]:
    try:
        session.sql(
            f"ALTER TABLE {FEATURE_VERSIONS_TABLE} ADD COLUMN IF NOT EXISTS {col_def}"
        ).collect()
    except Exception:
        pass

# Mark previous versions as inactive for this feature table
session.sql(
    f"""
    UPDATE {FEATURE_VERSIONS_TABLE}
    SET IS_ACTIVE = FALSE
    WHERE FEATURE_TABLE_NAME = 'UNI_BOX_FEATURES'
      AND IS_ACTIVE = TRUE
"""
).collect()

# Insert new metadata row for current version
session.sql(
    f"""
    INSERT INTO {FEATURE_VERSIONS_TABLE} (
        FEATURE_VERSION_ID,
        FEATURE_TABLE_NAME,
        CREATED_AT,
        CREATED_BY,
        ROW_COUNT,
        FEATURE_COUNT,
        SOURCE_TABLE,
        FEATURE_SNAPSHOT_AT,
        DESCRIPTION,
        IS_ACTIVE
    )
    SELECT
        '{feature_version_id}' AS FEATURE_VERSION_ID,
        'UNI_BOX_FEATURES' AS FEATURE_TABLE_NAME,
        CURRENT_TIMESTAMP() AS CREATED_AT,
        CURRENT_USER() AS CREATED_BY,
        {feature_count} AS ROW_COUNT,
        {len(feature_columns)} AS FEATURE_COUNT,
        '{TRAIN_TABLE_CLEANED}' AS SOURCE_TABLE,
        CURRENT_TIMESTAMP() AS FEATURE_SNAPSHOT_AT,
        'Materialized from TRAIN_DATASET_CLEANED (script 02_feature_store_setup.py)' AS DESCRIPTION,
        TRUE AS IS_ACTIVE
"""
).collect()

print("✅ Feature version metadata registered")

# %% [markdown]
# ## 4. Materializar Features en Tabla (sin FeatureView)

# %%
print("\n" + "=" * 80)
print("🧱 MATERIALIZING FEATURES INTO TABLE (CTAS)")
print("=" * 80)

print(f"\n📝 Creating/Replacing table: {FEATURES_TABLE}")
feature_df.write.mode("overwrite").save_as_table(FEATURES_TABLE)
print("✅ Features table created (without Dynamic Tables / Feature Views)")

print("\n📊 Sample of features (5 rows):")
session.table(FEATURES_TABLE).limit(5).show()

# %% [markdown]
# ## 6. Summary

# %%
print("\n" + "=" * 80)
print("✅ FEATURE DATASET SETUP COMPLETE!")
print("=" * 80)

print("\n📋 Summary:")
print(f"   ✅ Feature schema: {FEATURES_SCHEMA}")
print(f"   ✅ Features table: {FEATURES_TABLE}")
# Count actual feature columns (excluding metadata columns)
# METADATA_COLS is already in UPPER CASE
metadata_cols_upper = {col for col in METADATA_COLS}
actual_feature_count = len([col for col in feature_df.columns if col.upper() not in metadata_cols_upper])
print(f"   ✅ Features: {actual_feature_count} features")
print(f"   ✅ Excluded from features: {', '.join(EXCLUDED_COLS)}")
print(f"   ✅ Total records: {feature_count:,}")

print("\n💡 Next Steps:")
print("   1. Run 03_hyperparameter_search.py (will continue working without FeatureView)")
print("   2. Run 04_many_model_training.py (adjusted to not depend on FeatureView)")

print("\n" + "=" * 80)
