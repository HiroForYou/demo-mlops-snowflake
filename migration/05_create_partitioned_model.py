# %% [markdown]
# # Partitioned Model (16 models from 04)
#
# Loads the 16 group-specific models from the Model Registry (PRODUCTION alias),
# wraps them in a CustomModel with the partitioned API, and registers the
# combined model as UNI_BOX_REGRESSION_PARTITIONED.

# %% [markdown]
# ## 1. Setup

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F
from snowflake.ml.registry import Registry
from snowflake.ml.model import custom_model, task
from datetime import datetime
import pandas as pd
import numpy as np

session = get_active_session()

# %% [markdown]
# ### 1A. Constants

# %%
DATABASE        = "BD_AA_DEV"
STORAGE_SCHEMA  = "SC_STORAGE_BMX_PS"
MODELS_SCHEMA   = "SC_MODELS_BMX"
TRAIN_TABLE_CLEANED   = f"{DATABASE}.{STORAGE_SCHEMA}.TRAIN_DATASET_CLEANED"
PARTITIONED_MODEL_NAME = "UNI_BOX_REGRESSION_PARTITIONED"
VERSION_DATE           = datetime.now().strftime("%Y%m%d_%H%M")

STATS_NTILE_GROUP_COL = "STATS_NTILE_GROUP"

# Metadata / identifier columns excluded from the feature set
EXCLUDED_COLS = [
    "CUSTOMER_ID", "BRAND_PRES_RET", "PROD_KEY",
    "WEEK", STATS_NTILE_GROUP_COL,
]

session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()
registry = Registry(session=session, database_name=DATABASE, schema_name=MODELS_SCHEMA)
print(f"Session: {session.get_current_database()}.{session.get_current_schema()}")

# %% [markdown]
# ## 2. Load 16 Models (PRODUCTION alias)

# %%
groups_list = [
    row["GROUP_NAME"]
    for row in session.sql(f"""
        SELECT DISTINCT {STATS_NTILE_GROUP_COL} AS GROUP_NAME
        FROM {TRAIN_TABLE_CLEANED}
        WHERE {STATS_NTILE_GROUP_COL} IS NOT NULL
        ORDER BY {STATS_NTILE_GROUP_COL}
    """).collect()
]
print(f"Groups: {len(groups_list)}")

loaded_models = {}
feature_cols  = None

for group_name in groups_list:
    model_name = f"uni_box_regression_{group_name.lower()}"
    try:
        model_ref    = registry.get_model(model_name)
        model_version = model_ref.version("PRODUCTION")
        native_model  = model_version.load()
        if feature_cols is None:
            feature_cols = getattr(native_model, "feature_cols", None)
            if not feature_cols:
                raise ValueError("Model has no feature_cols attribute — run 04 first")
        loaded_models[group_name] = {"model": native_model, "model_version": model_version}
        ver_name = getattr(model_version, "name", str(model_version))
        print(f"  {group_name} -> version: {ver_name}")
    except Exception as e:
        print(f"  {group_name}: {str(e)[:100]}")

if not loaded_models:
    raise ValueError("No models loaded. Run 04_many_model_training.py first.")
print(f"\n{len(loaded_models)}/{len(groups_list)} models loaded  |  {len(feature_cols)} features")

# %% [markdown]
# ## 3. Define Partitioned Model Class

# %%
class PartitionedUniBoxModel(custom_model.CustomModel):
    """Routes predictions to the correct group sub-model via STATS_NTILE_GROUP.

    Wraps the 16 group-specific regressors trained in script 04.  Snowflake
    partitions input data by STATS_NTILE_GROUP before calling ``predict``, so
    each invocation always receives a single group value.
    """

    def __init__(self, model_context):
        """Initialise with a ModelContext holding each group sub-model.

        Parameters
        ----------
        model_context : snowflake.ml.model.custom_model.ModelContext
            Context keyed by lower-cased group name.
        """
        super().__init__(model_context)
        self.feature_cols = feature_cols

    @custom_model.partitioned_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for one STATS_NTILE_GROUP partition.

        Parameters
        ----------
        input_df : pandas.DataFrame
            Partition data; must include feature columns and context columns
            (CUSTOMER_ID, WEEK, BRAND_PRES_RET, PROD_KEY, STATS_NTILE_GROUP).

        Returns
        -------
        pandas.DataFrame
            Columns: CUSTOMER_ID, STATS_NTILE_GROUP, WEEK, BRAND_PRES_RET,
            PROD_KEY, predicted_uni_box_week.
        """
        if len(input_df) == 0:
            return pd.DataFrame(columns=[
                "CUSTOMER_ID", "STATS_NTILE_GROUP", "WEEK",
                "BRAND_PRES_RET", "PROD_KEY", "predicted_uni_box_week",
            ])

        part_col = STATS_NTILE_GROUP_COL
        if part_col not in input_df.columns:
            raise ValueError(f"Missing column '{part_col}'. Available: {list(input_df.columns)}")
        group_name = input_df[part_col].iloc[0]

        try:
            model = self.context.model_ref(group_name.lower())
        except Exception:
            raise ValueError(f"No model for group '{group_name}'. Keys: {list(self.context.models.keys())}")

        X       = input_df[self.feature_cols].fillna(0).astype(np.float64, errors="ignore")
        pred_out = model.predict(X)

        if isinstance(pred_out, pd.DataFrame):
            pred_col = next(
                (c for c in pred_out.columns if "PREDICT" in c.upper() or "OUTPUT" in c.upper()),
                pred_out.columns[0],
            )
            predictions = np.asarray(pred_out[pred_col], dtype=np.float64).ravel()
        else:
            predictions = np.asarray(pred_out).ravel()

        required_ctx = ["CUSTOMER_ID", "WEEK", "BRAND_PRES_RET", "PROD_KEY"]
        missing_ctx  = [c for c in required_ctx if c not in input_df.columns]
        if missing_ctx:
            raise ValueError(f"Missing context columns: {missing_ctx}")

        return pd.DataFrame({
            "CUSTOMER_ID":       input_df["CUSTOMER_ID"].values,
            STATS_NTILE_GROUP_COL: group_name,
            "WEEK":              input_df["WEEK"].values,
            "BRAND_PRES_RET":    input_df["BRAND_PRES_RET"].values,
            "PROD_KEY":          input_df["PROD_KEY"].values,
            "predicted_uni_box_week": predictions,
        })


# %% [markdown]
# ## 4. Build Model Context and Partitioned Model

# %%
models_dict = {gn.lower(): info["model"] for gn, info in loaded_models.items()}
model_context    = custom_model.ModelContext(models=models_dict)
partitioned_model = PartitionedUniBoxModel(model_context=model_context)
print(f"Partitioned model created with {len(models_dict)} sub-models")

# %% [markdown]
# ## 5. Prepare Sample Input

# %%
# The model was trained on feature_cols, so the sample must provide exactly those.
feat_cols_for_sample = feature_cols
training_df = session.table(TRAIN_TABLE_CLEANED)

sample_input_sp = (
    training_df
    .select(
        "CUSTOMER_ID", 
        STATS_NTILE_GROUP_COL, 
        "WEEK", 
        "BRAND_PRES_RET", 
        "PROD_KEY",
        *[F.col(c) for c in feat_cols_for_sample],
    )
    .filter(F.col(STATS_NTILE_GROUP_COL).isin(list(loaded_models.keys())))
    .group_by(STATS_NTILE_GROUP_COL)
    .agg(
        F.min("CUSTOMER_ID").alias("CUSTOMER_ID"),
        F.coalesce(F.min("WEEK"),          F.lit("000000")).alias("WEEK"),
        F.coalesce(F.min("BRAND_PRES_RET"), F.lit("UNKNOWN")).alias("BRAND_PRES_RET"),
        F.coalesce(F.min("PROD_KEY"),       F.lit("UNKNOWN")).alias("PROD_KEY"),
        *[F.min(F.col(c)).alias(c) for c in feat_cols_for_sample],
    )
    .select("CUSTOMER_ID", STATS_NTILE_GROUP_COL, "WEEK", "BRAND_PRES_RET", "PROD_KEY",
            *feat_cols_for_sample)
    .limit(min(16, len(loaded_models)))
)

if sample_input_sp.count() == 0:
    raise ValueError("Sample input is empty — verify TRAIN_DATASET_CLEANED groups match loaded models")

sample_input = sample_input_sp.to_pandas()
for col in ("WEEK", "BRAND_PRES_RET", "PROD_KEY"):
    sample_input[col] = sample_input[col].fillna("UNKNOWN" if col != "WEEK" else "000000")
for col in ("CUSTOMER_ID", STATS_NTILE_GROUP_COL, "WEEK", "BRAND_PRES_RET", "PROD_KEY"):
    sample_input[col] = sample_input[col].astype(str)
print(f"Sample input: {len(sample_input)} rows  |  {len(feat_cols_for_sample)} features")

# %% [markdown]
# ## 6. Register Partitioned Model

# %%
print(f"Registering {PARTITIONED_MODEL_NAME} v_{VERSION_DATE} ...")

# Collect the concrete version name for each sub-model (resolves the PRODUCTION alias)
import json as _json
submodel_versions = {}
for gn, info in loaded_models.items():
    mv_obj = info["model_version"]
    ver_name = getattr(mv_obj, "version_name", getattr(mv_obj, "name", str(mv_obj)))
    submodel_versions[gn] = ver_name

# Build the metrics dict: summary fields + one entry per sub-model version
partitioned_metrics = {
    "num_groups":       len(loaded_models),
    "num_features":     len(feature_cols),
    "model_type":       "mixed",
    "groups":           ",".join(sorted(loaded_models.keys())),
    # JSON snapshot of all sub-model versions used in this partitioned model
    "submodel_versions": _json.dumps(submodel_versions),
}
# Also store each sub-model version as an individual metric for easy filtering in Snowsight
for gn, ver in submodel_versions.items():
    partitioned_metrics[f"submodel_version_{gn}"] = ver

mv = registry.log_model(
    partitioned_model,
    model_name=PARTITIONED_MODEL_NAME,
    version_name=f"v_{VERSION_DATE}",
    comment=(
        f"Partitioned regression model for uni_box_week — "
        f"{len(loaded_models)} group-specific models (LGBM/XGB)"
    ),
    metrics=partitioned_metrics,
    sample_input_data=sample_input,
    task=task.Task.TABULAR_REGRESSION,
    options={"function_type": "TABLE_FUNCTION"},
)
print(f"Registered: {PARTITIONED_MODEL_NAME} v_{VERSION_DATE}")
print("Sub-model versions baked in:")
for gn, ver in sorted(submodel_versions.items()):
    print(f"  {gn}: {ver}")

model_fqn = f"{DATABASE}.{MODELS_SCHEMA}.{PARTITIONED_MODEL_NAME}"
try:
    session.sql(f"ALTER MODEL {model_fqn} VERSION PRODUCTION UNSET ALIAS").collect()
    print("Previous PRODUCTION alias removed")
except Exception:
    pass
session.sql(f"ALTER MODEL {model_fqn} VERSION v_{VERSION_DATE} SET ALIAS=PRODUCTION").collect()
print("PRODUCTION alias assigned to new version")

# %% [markdown]
# ## 7. Verify Registration

# %%
result = session.sql(f"""
    SHOW MODELS LIKE '{PARTITIONED_MODEL_NAME}' IN SCHEMA {DATABASE}.{MODELS_SCHEMA}
""").collect()

if result:
    versions = session.sql(f"""
        SHOW VERSIONS IN MODEL {DATABASE}.{MODELS_SCHEMA}.{PARTITIONED_MODEL_NAME}
    """).collect()
    print(f"Model found in registry — {len(versions)} version(s):")
    for v in versions[-3:]:
        print(f"  - {v['name']}")
else:
    print("Model not found in registry")

# %% [markdown]
# ## 8. Summary

# %%
print(f"\nPartitioned model ready:")
print(f"  Name:     {PARTITIONED_MODEL_NAME}")
print(f"  Version:  v_{VERSION_DATE}")
print(f"  Alias:    PRODUCTION")
print(f"  Groups:   {len(loaded_models)} ({', '.join(sorted(loaded_models.keys())[:4])}...)")
print(f"  Features: {len(feature_cols)}")
print("\nNext: 06_create_baselines.py")
