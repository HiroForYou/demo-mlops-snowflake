# %% [markdown]
# # Partitioned Model (16 models from 04)
# Loads group-specific models from Registry, creates CustomModel with partitioned API and registers it.

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F
from snowflake.ml.registry import Registry
from snowflake.ml.model import custom_model, task
import pandas as pd
import numpy as np
from datetime import datetime

session = get_active_session()

# Configuration: Database, schemas, and tables
DATABASE = "BD_AA_DEV"
STORAGE_SCHEMA = "SC_STORAGE_BMX_PS"
MODELS_SCHEMA = "SC_MODELS_BMX"
TRAIN_TABLE_CLEANED = f"{DATABASE}.{STORAGE_SCHEMA}.TRAIN_DATASET_CLEANED"
PARTITIONED_MODEL_NAME = "UNI_BOX_REGRESSION_PARTITIONED"

# Column constants
STATS_NTILE_GROUP_COL = "STATS_NTILE_GROUP"

# Excluded columns (metadata columns, not features) - defined once at the beginning
EXCLUDED_COLS = [
    "CUSTOMER_ID",
    "BRAND_PRES_RET",
    "PROD_KEY",
    "WEEK",
    "FEATURE_TIMESTAMP",
    STATS_NTILE_GROUP_COL,
]

session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()
registry = Registry(session=session, database_name=DATABASE, schema_name=MODELS_SCHEMA)
print(f"✅ {session.get_current_database()}.{session.get_current_schema()}")

# %% [markdown]
# ## 1. Load the 16 Models (PRODUCTION)

# %%

groups_df = session.sql(
    f"""
    SELECT DISTINCT {STATS_NTILE_GROUP_COL} AS GROUP_NAME
    FROM {TRAIN_TABLE_CLEANED}
    WHERE {STATS_NTILE_GROUP_COL} IS NOT NULL
    ORDER BY {STATS_NTILE_GROUP_COL}
"""
)
groups_list = [row["GROUP_NAME"] for row in groups_df.collect()]
print(f"\n📊 Groups: {len(groups_list)}")

loaded_models = {}
feature_cols = None

for group_name in groups_list:
    model_name = f"uni_box_regression_{group_name.lower()}"
    try:
        model_ref = registry.get_model(model_name)
        model_version = model_ref.version("PRODUCTION")
        native_model = model_version.load()
        if feature_cols is None:
            feature_cols = getattr(native_model, "feature_cols", None)
            if not feature_cols:
                raise ValueError("Model has no feature_cols; run 04 first.")
        loaded_models[group_name] = {"model": native_model, "model_version": model_version, "model_name": model_name}
        ver_name = getattr(model_version, "name", str(model_version))
        print(f"✅ {group_name} (version: {ver_name})")
    except Exception as e:
        print(f"❌ {group_name}: {str(e)[:100]}")

if not loaded_models:
    raise ValueError("No models loaded. Run 04_many_model_training.py first.")
if len(loaded_models) < len(groups_list):
    print(f"⚠️  {len(loaded_models)}/{len(groups_list)} models loaded")
# Collect PRODUCTION versions in a readable way (library versions may expose different attributes)
versions_loaded = {
    getattr(m["model_version"], "version", getattr(m["model_version"], "version_name", str(m["model_version"])))
    for m in loaded_models.values()
}
print(f"\n✅ {len(loaded_models)} models, {len(feature_cols)} features (alias PRODUCTION → {versions_loaded or 'N/A'})")

# %% [markdown]
# ## 2. Define Partitioned Model Class

# %%
print("\n" + "=" * 80)
print("🔧 DEFINING PARTITIONED MODEL CLASS")
print("=" * 80)


class PartitionedUniBoxModel(custom_model.CustomModel):
    """Partitioned model: routes predictions by STATS_NTILE_GROUP."""

    def __init__(self, model_context):
        super().__init__(model_context)
        # Use the same feature columns detected when loading base models
        self.feature_cols = feature_cols

    @custom_model.partitioned_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Prediction per group; routes according to STATS_NTILE_GROUP."""
        if len(input_df) == 0:
            return pd.DataFrame(
                columns=[
                    "CUSTOMER_ID",
                    "STATS_NTILE_GROUP",
                    "WEEK",
                    "BRAND_PRES_RET",
                    "PROD_KEY",
                    "predicted_uni_box_week",
                ]
            )

        # Single standard: UPPERCASE (avoids case-insensitive duplicates in signature)
        part_col = STATS_NTILE_GROUP_COL
        if part_col not in input_df.columns:
            raise ValueError(f"Missing required column '{part_col}' in input_df. Columns: {list(input_df.columns)}")
        group_name = input_df[part_col].iloc[0]

        model_key = group_name.lower()
        try:
            model = self.context.model_ref(model_key)
        except Exception:
            try:
                model = self.context.model_ref(group_name.lower())
            except Exception:
                raise ValueError(f"Model not found for group: {group_name}. Keys: {list(self.context.models.keys())}")

        # Select only the known feature columns
        X = input_df[self.feature_cols].fillna(0).astype(np.float64, errors="ignore")
        pred_out = model.predict(X)

        if isinstance(pred_out, pd.DataFrame):
            pred_col = next((c for c in pred_out.columns if "PREDICT" in c.upper() or "OUTPUT" in c.upper()), pred_out.columns[0])
            predictions = np.asarray(pred_out[pred_col], dtype=np.float64).ravel()
        elif hasattr(pred_out, "flatten"):
            predictions = np.asarray(pred_out).flatten()
        else:
            predictions = np.asarray(pred_out).ravel()

        # Propagate context columns to avoid ambiguous JOINs in inference
        required_ctx = ["CUSTOMER_ID", "WEEK", "BRAND_PRES_RET", "PROD_KEY"]
        missing_ctx = [c for c in required_ctx if c not in input_df.columns]
        if missing_ctx:
            raise ValueError(f"Missing required context columns: {missing_ctx}. Columns: {list(input_df.columns)}")

        cust = input_df["CUSTOMER_ID"].values
        week_vals = input_df["WEEK"].values
        brand_vals = input_df["BRAND_PRES_RET"].values
        prod_key_vals = input_df["PROD_KEY"].values

        return pd.DataFrame(
            {
                "CUSTOMER_ID": cust,
                STATS_NTILE_GROUP_COL: group_name,
                "WEEK": week_vals,
                "BRAND_PRES_RET": brand_vals,
                "PROD_KEY": prod_key_vals,
                "predicted_uni_box_week": predictions,
            }
        )


print("✅ PartitionedUniBoxModel class defined")

# %% [markdown]
# ## 3. Create Model Context and Partitioned Model

# %%
print("\n" + "=" * 80)
print("📦 CREATING PARTITIONED MODEL")
print("=" * 80)

models_dict = {}
for group_name, model_info in loaded_models.items():
    model_key = group_name.lower()
    models_dict[model_key] = model_info["model"]
    print(f"   Added {group_name} → {model_key}")
print(f"\n✅ ModelContext created with {len(models_dict)} models")

model_context = custom_model.ModelContext(models=models_dict)
partitioned_model = PartitionedUniBoxModel(model_context=model_context)
print("✅ Partitioned model created")

# %% [markdown]
# ## 4. Sample Input (aligned with 06: includes week and brand_pres_ret as context)

# %%
print("\n" + "=" * 80)
print("📋 PREPARING SAMPLE INPUT (ALIGNED WITH 06)")
print("=" * 80)

# Use the same exclusion list as 02/04/06 to ensure sample_input has the same signature as inference
# EXCLUDED_COLS is already in UPPER CASE
excluded_cols_with_target = EXCLUDED_COLS + ["UNI_BOX_WEEK"]
excluded_upper_sample = {col for col in excluded_cols_with_target}

# Get training schema to identify numeric features (same as 06 with inference)
training_schema = session.sql(f"DESCRIBE TABLE {TRAIN_TABLE_CLEANED}").collect()
col_type_dict_sample = {row["name"].upper(): str(row["type"]).upper() for row in training_schema}
all_cols_sample = [row["name"] for row in training_schema]

NUMERIC_PREFIXES_SAMPLE = ("FLOAT", "NUMBER", "INTEGER", "BIGINT", "DOUBLE")
feature_cols_sample = [
    c for c in all_cols_sample
    if c.upper() not in excluded_upper_sample
    and (col_type_dict_sample.get(c.upper()) or "").startswith(NUMERIC_PREFIXES_SAMPLE)
]

# Verify that feature_cols_sample matches feature_cols from models (should be equal)
if set(c.upper() for c in feature_cols_sample) != set(c.upper() for c in feature_cols):
    print(f"⚠️  Feature mismatch: sample has {len(feature_cols_sample)}, models have {len(feature_cols)}")
    print(f"   Using feature_cols from models to maintain compatibility")
    feature_cols_for_sample = feature_cols
else:
    # Sort feature_cols_sample according to feature_cols order to maintain consistency
    feature_cols_for_sample = [c for c in feature_cols if c.upper() in {x.upper() for x in feature_cols_sample}]
    if len(feature_cols_for_sample) != len(feature_cols):
        print(f"⚠️  Some model features are not in sample, using all model features")
        feature_cols_for_sample = feature_cols

training_df = session.table(TRAIN_TABLE_CLEANED)

# Detect actual names (case-insensitive) to avoid NULL columns due to casing (WEEK vs week)
cust_col = next((c for c in training_df.columns if c.upper() == "CUSTOMER_ID"), "CUSTOMER_ID")
part_col = next((c for c in training_df.columns if c.upper() == STATS_NTILE_GROUP_COL), STATS_NTILE_GROUP_COL)
week_col = next((c for c in training_df.columns if c.upper() == "WEEK"), "WEEK")
brand_col = next((c for c in training_df.columns if c.upper() == "BRAND_PRES_RET"), "BRAND_PRES_RET")
prod_key_col = next((c for c in training_df.columns if c.upper() == "PROD_KEY"), "PROD_KEY")

# Select and alias context columns with fixed names for model signature
base_selected = training_df.select(
    # Single standard: UPPERCASE
    F.col(cust_col).alias("CUSTOMER_ID"),
    F.col(part_col).alias(STATS_NTILE_GROUP_COL),
    # Use UPPERCASE to avoid case-insensitive collisions in signature inference
    F.col(week_col).alias("WEEK"),
    F.col(brand_col).alias("BRAND_PRES_RET"),
    F.col(prod_key_col).alias("PROD_KEY"),
    *[F.col(c) for c in feature_cols_for_sample],
)

sample_input_sp = (
    base_selected.filter(F.col(STATS_NTILE_GROUP_COL).isin(list(loaded_models.keys())))
    .group_by(STATS_NTILE_GROUP_COL)
    .agg(
        # Ensure NO-NULL in context columns to infer signature (Snowflake ML requires at least one non-null)
        F.min(F.col("CUSTOMER_ID")).alias("CUSTOMER_ID"),
        F.coalesce(F.min(F.col("WEEK")), F.lit("000000")).alias("WEEK"),
        F.coalesce(F.min(F.col("BRAND_PRES_RET")), F.lit("UNKNOWN")).alias("BRAND_PRES_RET"),
        F.coalesce(F.min(F.col("PROD_KEY")), F.lit("UNKNOWN")).alias("PROD_KEY"),
        *[F.min(F.col(c)).alias(c) for c in feature_cols_for_sample],
    )
    .select("CUSTOMER_ID", STATS_NTILE_GROUP_COL, "WEEK", "BRAND_PRES_RET", "PROD_KEY", *feature_cols_for_sample)
    .limit(min(16, len(loaded_models)))
)
# Count first to verify we have data (using Snowpark, no pandas conversion yet)
sample_count = sample_input_sp.count()
if sample_count == 0:
    raise ValueError(f"Sample input is empty. Verify that TRAIN_DATASET_CLEANED has the same {STATS_NTILE_GROUP_COL} as loaded models.")

# Only convert to pandas when necessary for model registration (small sample, minimal memory)
sample_input = sample_input_sp.to_pandas()
print(f"✅ Sample input prepared: {len(sample_input)} rows (one per group)")
print(f"   Columns: CUSTOMER_ID, {STATS_NTILE_GROUP_COL}, WEEK, BRAND_PRES_RET, PROD_KEY, {len(feature_cols_for_sample)} features")

# Final safeguard in pandas (UPPERCASE only; no extra columns created)
sample_input["WEEK"] = sample_input["WEEK"].fillna("000000")
sample_input["BRAND_PRES_RET"] = sample_input["BRAND_PRES_RET"].fillna("UNKNOWN")
sample_input["PROD_KEY"] = sample_input["PROD_KEY"].fillna("UNKNOWN")

# Force context types to string so model signature expects VARCHAR (and matches inference).
# (In training WEEK usually comes numeric; if NUMBER is inferred here, it fails when inference receives VARCHAR.)
sample_input["CUSTOMER_ID"] = sample_input["CUSTOMER_ID"].astype(str)
sample_input[STATS_NTILE_GROUP_COL] = sample_input[STATS_NTILE_GROUP_COL].astype(str)
sample_input["WEEK"] = sample_input["WEEK"].astype(str)
sample_input["BRAND_PRES_RET"] = sample_input["BRAND_PRES_RET"].astype(str)
sample_input["PROD_KEY"] = sample_input["PROD_KEY"].astype(str)

# Quick debug (will be seen in notebook/script execution)
try:
    null_week = int(sample_input["WEEK"].isna().sum())
    null_brand = int(sample_input["BRAND_PRES_RET"].isna().sum())
    null_prod_key = int(sample_input["PROD_KEY"].isna().sum())
    print(f"🔎 sample_input nulls — week: {null_week}, brand_pres_ret: {null_brand}, prod_key: {null_prod_key}")
except Exception:
    pass

# %% [markdown]
# ## 5. Register Partitioned Model

# %%
print("\n" + "=" * 80)
print("📝 REGISTERING PARTITIONED MODEL")
print("=" * 80)

version_date = datetime.now().strftime("%Y%m%d_%H%M")

print(f"\n📝 Registering in Model Registry...")
print(f"   Name: {PARTITIONED_MODEL_NAME}")
print(f"   Version: v_{version_date}")

try:
    mv = registry.log_model(
        partitioned_model,
        model_name=PARTITIONED_MODEL_NAME,
        version_name=f"v_{version_date}",
        comment=f"Partitioned regression model for uni_box_week - Combines {len(loaded_models)} group-specific models (LGBM/XGB/SGD)",
        metrics={
            "num_groups": len(loaded_models),
            "num_features": len(feature_cols),
            "model_type": "mixed",
            "groups": ",".join(sorted(loaded_models.keys())),
        },
        sample_input_data=sample_input,
        task=task.Task.TABULAR_REGRESSION,
        options={"function_type": "TABLE_FUNCTION"},
    )

    print("\n✅ Partitioned model registered successfully!")
    # Move PRODUCTION alias in a "clean" way:
    # 1) remove alias from previous version (if exists)
    # 2) assign alias to new version
    model_fqn = f"{DATABASE}.{MODELS_SCHEMA}.{PARTITIONED_MODEL_NAME}"
    # NOTE: the version_name logged above is "v_<timestamp>" (lowercase).
    # We use exactly that identifier to avoid ambiguity.
    new_version_name = f"v_{version_date}"
    try:
        session.sql(f"ALTER MODEL {model_fqn} VERSION PRODUCTION UNSET ALIAS").collect()
        print("🧹 Alias 'PRODUCTION' removed from previous version")
    except Exception as alias_unset_err:
        print(f"ℹ️  Could not remove previous alias (may not exist): {str(alias_unset_err)[:120]}")

    session.sql(
        # Alias is an IDENTIFIER; avoid single quotes to prevent parsing errors.
        f"ALTER MODEL {model_fqn} VERSION {new_version_name} SET ALIAS=PRODUCTION"
    ).collect()
    print("🏷️  Alias 'PRODUCTION' moved to new version")

except Exception as e:
    print(f"\n❌ Error registering model: {str(e)}")
    raise

# %% [markdown]
# ## 6. Verify Registration

# %%
print("\n" + "=" * 80)
print("🔍 VERIFYING REGISTRATION")
print("=" * 80)

result = session.sql(
    f"""
    SHOW MODELS LIKE '{PARTITIONED_MODEL_NAME}' 
    IN SCHEMA {DATABASE}.{MODELS_SCHEMA}
"""
).collect()

if result:
    print("✅ Partitioned model found in registry")

    versions = session.sql(
        f"""
        SHOW VERSIONS IN MODEL {DATABASE}.{MODELS_SCHEMA}.{PARTITIONED_MODEL_NAME}
    """
    ).collect()

    print(f"\n📊 Versions: {len(versions)}")
    for v in versions[-3:]:
        print(f"   - {v['name']}")
else:
    print("❌ Model not found in registry")

# %% [markdown]
# ## 7. Summary

# %%
print("\n" + "=" * 80)
print("✅ PARTITIONED MODEL CREATION COMPLETE!")
print("=" * 80)

print("\n📋 Summary:")
print(f"   ✅ Source models: {len(loaded_models)} group-specific models")
print(f"   ✅ Partitioned model: {PARTITIONED_MODEL_NAME}")
print(f"   ✅ Version: v_{version_date}")
print(f"   ✅ Alias: PRODUCTION")
print(f"   ✅ Features: {len(feature_cols)}")
print(
    f"   ✅ Groups: {', '.join(sorted(loaded_models.keys())[:5])}... ({len(loaded_models)} total)"
)

print("\n💡 Next Steps:")
print("   1. Review partitioned model registration")
print("   2. Run 06_partitioned_inference_batch.py for batch inference")
print(
    "   3. Use partitioned inference syntax: TABLE(model!PREDICT(...) OVER (PARTITION BY stats_ntile_group))"
)

print("\n🎯 Key Benefits:")
print("   - 16 models combined into one partitioned model")
print("   - Automatic routing by stats_ntile_group")
print("   - Consistent inference syntax")
print("   - Each group uses its optimized hyperparameters")

print("\n" + "=" * 80)
