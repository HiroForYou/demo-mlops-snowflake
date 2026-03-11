# %% [markdown]
# # Feature Dataset Materialization
#
# Builds and materializes a feature dataset from TRAIN_DATASET_CLEANED into a
# plain Snowflake table.  Steps performed:
# 1. Ensure the destination schema exists.
# 2. Build the feature query (all columns except identifiers and target).
# 3. Materialize features into UNI_BOX_FEATURES (CTAS / overwrite).

# %% [markdown]
# ## 1. Setup

# %%
from snowflake.snowpark.context import get_active_session

session = get_active_session()

# %% [markdown]
# ### 1A. Constants

# %%
DATABASE        = "BD_AA_DEV"
STORAGE_SCHEMA  = "SC_STORAGE_BMX_PS"
FEATURES_SCHEMA = "SC_FEATURES_BMX"
TRAIN_TABLE_CLEANED = f"{DATABASE}.{STORAGE_SCHEMA}.TRAIN_DATASET_CLEANED"
FEATURES_TABLE      = f"{DATABASE}.{FEATURES_SCHEMA}.UNI_BOX_FEATURES"

TARGET_COLUMN       = "UNI_BOX_WEEK"
STATS_NTILE_GROUP_COL = "STATS_NTILE_GROUP"

# Metadata / identifier columns excluded from the feature set
EXCLUDED_COLS = [
    "CUSTOMER_ID",
    "BRAND_PRES_RET",
    "WEEK",
    STATS_NTILE_GROUP_COL,
    "PROD_KEY",
    TARGET_COLUMN,
]

session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()
print(f"Session: {session.get_current_database()}.{session.get_current_schema()}")

# %% [markdown]
# ## 2. Ensure Features Schema Exists

# %%
session.sql(f"USE SCHEMA {FEATURES_SCHEMA}").collect()
print(f"Schema ready: {FEATURES_SCHEMA}")

# %% [markdown]
# ## 3. Build Feature Query

# %%
columns_info = session.sql(f"DESCRIBE TABLE {TRAIN_TABLE_CLEANED}").collect()
all_columns  = [row["name"] for row in columns_info]

feature_columns = [col for col in all_columns if col not in EXCLUDED_COLS]

print(f"Total columns: {len(all_columns)}")
print(f"Feature columns: {len(feature_columns)}")

feature_cols_str = ",\n        ".join(feature_columns)

feature_df = session.sql(f"""
    SELECT
        CUSTOMER_ID,
        BRAND_PRES_RET,
        PROD_KEY,
        WEEK,
        {feature_cols_str}
    FROM {TRAIN_TABLE_CLEANED}
    WHERE CUSTOMER_ID IS NOT NULL
      AND BRAND_PRES_RET IS NOT NULL
""")

feature_count = feature_df.count()
print(f"Feature records: {feature_count:,}")

# %% [markdown]
# ## 4. Materialize Features (CTAS)

# %%
feature_df.write.mode("overwrite").save_as_table(FEATURES_TABLE)
print(f"Features table ready: {FEATURES_TABLE}")

print("\nSample (5 rows):")
session.table(FEATURES_TABLE).limit(5).show()

# %% [markdown]
# ## 5. Summary

# %%
print(f"Features table: {FEATURES_TABLE}")
print(f"Feature columns: {len(feature_df.columns)}")
print(f"Total records: {feature_count:,}")
print("\nNext: 03_hyperparameter_search.py")
