# %% [markdown]
# # Migration: Feature Store Setup
#
# ## Overview
# This script creates a Feature Store and defines FeatureViews for the uni_box_week regression model.
#
# ## What We'll Do:
# 1. Create Feature Store schema
# 2. Define Entity (if applicable)
# 3. Create FeatureView with features from cleaned tables
# 4. Register FeatureView in Feature Store

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.ml.feature_store import FeatureStore, Entity, FeatureView, CreationMode

session = get_active_session()

# Set context
session.sql("USE DATABASE BD_AA_DEV").collect()
session.sql("USE SCHEMA SC_STORAGE_BMX_PS").collect()

print(f"✅ Connected to Snowflake")
print(f"   Database: {session.get_current_database()}")
print(f"   Schema: {session.get_current_schema()}")

# %% [markdown]
# ## 1. Create Feature Store Schema

# %%
print("\n" + "=" * 80)
print("🏪 CREATING FEATURE STORE")
print("=" * 80)

# Create schema for Feature Store
session.sql("CREATE SCHEMA IF NOT EXISTS BD_AA_DEV.FEATURE_STORE").collect()
session.sql("USE SCHEMA BD_AA_DEV.FEATURE_STORE").collect()

print("\n✅ Feature Store schema created")

# Initialize Feature Store
fs = FeatureStore(
    session=session,
    database="BD_AA_DEV",
    name="FEATURE_STORE",
    creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
)

print("✅ Feature Store initialized")

# %% [markdown]
# ## 2. Define Entity (Optional)

# %%
print("\n" + "=" * 80)
print("👤 DEFINING ENTITIES")
print("=" * 80)

# Define Customer-Product entity (combination of customer_id and brand_pres_ret)
customer_product_entity = Entity(
    name="CUSTOMER_PRODUCT",
    join_keys=["customer_id", "brand_pres_ret"],
    desc="Customer-Product combination entity for uni_box_week regression",
)

try:
    fs.register_entity(customer_product_entity)
    print("✅ Customer-Product entity registered")
except Exception as e:
    print(f"⚠️  Entity may already exist: {str(e)[:100]}")

# %% [markdown]
# ## 3. Create FeatureView from Cleaned Training Data

# %%
print("\n" + "=" * 80)
print("📋 CREATING FEATURE VIEW")
print("=" * 80)

# Define excluded columns (not features)
excluded_cols = [
    "customer_id",
    "brand_pres_ret",
    "week",
    "group",
    "stats_group",
    "percentile_group",
    "stats_ntile_group",
]

# Create FeatureView query
# This selects all features (excluding excluded columns and target) from cleaned training data
feature_df = session.sql(
    """
    SELECT
        customer_id,
        brand_pres_ret,
        week,
        -- Temporal features
        week_of_year,
        -- Past sales features
        sum_past_12_weeks,
        avg_past_12_weeks,
        max_past_12_weeks,
        sum_past_24_weeks,
        avg_past_24_weeks,
        max_past_24_weeks,
        sum_past_4_weeks,
        avg_past_4_weeks,
        max_past_4_weeks,
        sum_p4w,
        -- Previous period features
        max_prev2,
        avg_prev2,
        max_prev3,
        avg_prev3,
        -- Weekly totals
        w_m1_total,
        w_m2_total,
        w_m3_total,
        w_m4_total,
        -- Store features
        num_coolers,
        num_doors,
        -- Category features
        pharm_super_conv,
        wines_liquor,
        groceries,
        spec_foods,
        -- Other features
        avg_avg_daily_all_hours,
        prod_key,
        -- Timestamp for Feature Store
        CASE 
            WHEN week IS NOT NULL THEN 
                TRY_TO_TIMESTAMP_NTZ(week, 'YYYYWW')
            ELSE CURRENT_TIMESTAMP()
        END AS FEATURE_TIMESTAMP
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED
    WHERE customer_id IS NOT NULL
        AND brand_pres_ret IS NOT NULL
"""
)

print("✅ Feature query created")

# Count features
feature_count = feature_df.count()
print(f"   Total feature records: {feature_count:,}")

# %% [markdown]
# ## 4. Register FeatureView

# %%
print("\n📝 Registering FeatureView...")

# Create FeatureView
uni_box_feature_view = FeatureView(
    name="UNI_BOX_FEATURES",
    entities=[customer_product_entity],
    feature_df=feature_df,
    timestamp_col="FEATURE_TIMESTAMP",
    refresh_freq="1 day",  # Adjust based on your needs
    desc="Features for uni_box_week regression model - includes temporal, past sales, store, and category features",
)

# Register FeatureView
try:
    registered_fv = fs.register_feature_view(
        feature_view=uni_box_feature_view, version="v1", block=True
    )
    print("✅ FeatureView 'UNI_BOX_FEATURES' registered successfully")
    print(f"   Version: v1")
    print(f"   Refresh frequency: 1 day")
except Exception as e:
    error_msg = str(e)
    if "already exists" in error_msg.lower():
        print(f"⚠️  FeatureView already exists. Updating...")
        # Try to update or create new version
        try:
            registered_fv = fs.register_feature_view(
                feature_view=uni_box_feature_view, version="v2", block=True
            )
            print("✅ FeatureView registered as v2")
        except Exception as e2:
            print(f"❌ Error updating FeatureView: {str(e2)[:200]}")
            raise
    else:
        print(f"❌ Error registering FeatureView: {error_msg[:200]}")
        raise

# %% [markdown]
# ## 5. Verify FeatureView

# %%
print("\n" + "=" * 80)
print("🔍 VERIFYING FEATURE VIEW")
print("=" * 80)

# List registered FeatureViews
try:
    feature_views = fs.list_feature_views()
    print(f"\n📋 Registered FeatureViews ({len(feature_views)}):")
    for fv in feature_views:
        print(f"   - {fv.name} (version: {fv.version})")
except Exception as e:
    print(f"⚠️  Could not list FeatureViews: {str(e)[:100]}")

# Sample features
print("\n📊 Sample Feature Data:")
sample_features = feature_df.limit(5)
sample_features.show()

# %% [markdown]
# ## 6. Summary

# %%
print("\n" + "=" * 80)
print("✅ FEATURE STORE SETUP COMPLETE!")
print("=" * 80)

print("\n📋 Summary:")
print(f"   ✅ Feature Store: BD_AA_DEV.FEATURE_STORE")
print(f"   ✅ Entity: CUSTOMER_PRODUCT")
print(f"   ✅ FeatureView: UNI_BOX_FEATURES (v1)")
print(
    f"   ✅ Features: {len([col for col in feature_df.columns if col not in ['customer_id', 'brand_pres_ret', 'week', 'FEATURE_TIMESTAMP']])} features"
)
print(f"   ✅ Total records: {feature_count:,}")

print("\n💡 Next Steps:")
print("   1. Review FeatureView definition")
print("   2. Run 03_hyperparameter_search.py to find optimal hyperparameters")
print("   3. Features can be materialized for training when needed")

print("\n" + "=" * 80)
