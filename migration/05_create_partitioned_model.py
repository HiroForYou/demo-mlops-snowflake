# %% [markdown]
# # Partitioned Model (16 models from 04)
#
# Loads the 16 group-specific models from the Model Registry (PRODUCTION alias),
# wraps them in a PartitionedModel with the partitioned API, and registers the
# combined model as UNIBOX_CUSTBPR_WEEKLY_FORECAST.

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
FEATURES_SCHEMA = "SC_FEATURES_BMX"
MODELS_SCHEMA   = "SC_MODELS_BMX"

# Model name (base for all model-specific objects)
MODEL_NAME = "UNIBOX_CUSTBPR_WEEKLY_FORECAST"

# Feature store name (shared by entity and frequency, not tied to model)
FEATURE_STORE_NAME = "FEAT_CUSTBPR_WEEKLY"

# Input table
TRAIN_TABLE_CLEANED = f"{DATABASE}.{FEATURES_SCHEMA}.{FEATURE_STORE_NAME}__TRAIN"

# Version date
VERSION_DATE = datetime.now().strftime("%Y%m%d_%H%M")

# Partition, prediction and context columns
STATS_NTILE_GROUP_COL = "STATS_NTILE_GROUP"
PREDICTION_COL = "predicted_uni_box_week"
CONTEXT_COLS   = ["CUSTOMER_ID", "WEEK", "BRAND_PRES_RET", "PROD_KEY"]


session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()
registry = Registry(session=session, database_name=DATABASE, schema_name=MODELS_SCHEMA)
print(f"Session: {session.get_current_database()}.{session.get_current_schema()}")

# %% [markdown]
# ## 2. Load 16 Models (PRODUCTION alias)
#
# Loads the already-trained group-specific regressors from the Model
# Registry using the `PRODUCTION` alias, and captures the feature schema
# expected by the native models.

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
    model_name = f"{MODEL_NAME.lower()}__{group_name.lower()}"
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
#
# Implements a `CustomModel` with the partitioned API. The `predict()` method
# routes incoming rows to the correct partition-specific sub-model based on a
# partition column (e.g. `STATS_NTILE_GROUP_COL`), and returns a unified
# prediction table with configurable context + output columns.

# %%
class PartitionedModel(custom_model.CustomModel):
    """Generic partition router for a registry-loaded set of sub-models.

    Snowflake partitions the input data by `partition_col` before calling
    ``predict`` (via ``@custom_model.partitioned_api``), so each invocation
    should receive exactly one partition value.

    The class is intentionally model-agnostic:
    - it does not hardcode column names (context/output/partition),
    - it accepts the feature list externally (to match any trained schema),
    - it tolerates messy/object pandas inputs by coercing features to numeric
      with ``errors="coerce"`` (keeping missing as NaN).
    """

    def __init__(
        self,
        model_context,
        *,
        feature_cols: list[str] | None = None,
        partition_col: str | None = None,
        context_cols: list[str] | None = None,
        prediction_col: str | None = None,
        model_key_fn=None,
    ):
        """Initialise with a ModelContext holding each partition sub-model.

        Parameters
        ----------
        model_context : snowflake.ml.model.custom_model.ModelContext
            Context holding sub-models, keyed by a derived partition key.
        feature_cols : list[str]
            Feature columns passed to the native sub-models.
        partition_col : str
            Column used by Snowflake to partition inputs (one value per call).
        context_cols : list[str]
            Context/identifier columns copied through to the output.
        prediction_col : str
            Name of the output prediction column produced by this wrapper.
        model_key_fn : callable, optional
            Function that maps the raw partition value to a ModelContext key.
            Defaults to ``lambda v: str(v).lower()``.
        """
        super().__init__(model_context)
        # Registry may re-instantiate CustomModel with only `model_context`,
        # so kwargs are optional and resolved from context/defaults.
        self.feature_cols    = list(feature_cols) if feature_cols is not None else self._infer_feature_cols_from_context()
        self.partition_col   = partition_col or STATS_NTILE_GROUP_COL
        self.context_cols    = list(context_cols) if context_cols is not None else list(CONTEXT_COLS)
        self.prediction_col  = prediction_col or PREDICTION_COL
        self.model_key_fn    = model_key_fn or (lambda v: str(v).lower())

    def _infer_feature_cols_from_context(self) -> list[str]:
        """Infer feature columns from the first sub-model in ModelContext."""
        try:
            models_map = getattr(self.context, "models", {}) or {}
            if not models_map:
                return []
            first_model = next(iter(models_map.values()))
            cols = getattr(first_model, "feature_cols", None)
            return list(cols) if cols else []
        except Exception:
            return []

    @custom_model.partitioned_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for one partition.

        Parameters
        ----------
        input_df : pandas.DataFrame
            Partition data; must include:
            - `partition_col`
            - `feature_cols`
            - `context_cols` (for passthrough output)

        Returns
        -------
        pandas.DataFrame
            Columns: `context_cols` + `partition_col` + `prediction_col`.
        """
        if len(input_df) == 0:
            return pd.DataFrame(columns=[*self.context_cols, self.partition_col, self.prediction_col])

        if not self.feature_cols:
            # Registry can instantiate the model without optional constructor kwargs during
            # validation/inference checks. Resolve features lazily from the provided input:
            # all columns except context + partition (+ output if present).
            excluded = set(self.context_cols + [self.partition_col, self.prediction_col])
            inferred = [c for c in input_df.columns if c not in excluded]
            if inferred:
                self.feature_cols = inferred
            else:
                # Final fallback: try to infer from context models if available.
                self.feature_cols = self._infer_feature_cols_from_context()
            if not self.feature_cols:
                raise ValueError(
                    "PartitionedModel could not resolve `feature_cols`. "
                    "Pass `feature_cols=` explicitly or provide input columns containing features."
                )

        if self.partition_col not in input_df.columns:
            raise ValueError(
                f"Missing partition column '{self.partition_col}'. Available: {list(input_df.columns)}"
            )
        partition_value = input_df[self.partition_col].iloc[0]

        try:
            model = self.context.model_ref(self.model_key_fn(partition_value))
        except Exception:
            raise ValueError(
                f"No model for partition '{partition_value}'. Keys: {list(self.context.models.keys())}"
            )

        missing_feats = [c for c in self.feature_cols if c not in input_df.columns]
        if missing_feats:
            raise ValueError(f"Missing feature columns: {missing_feats}")

        X_raw = input_df[self.feature_cols]
        # Defensive conversion: allow messy/object inputs without crashing, keep missing as NaN.
        X     = X_raw.apply(pd.to_numeric, errors="coerce").astype(np.float64)
        pred_out = model.predict(X)

        if isinstance(pred_out, pd.DataFrame):
            # Prefer the wrapper's declared output cols when available; otherwise
            # try common naming conventions.
            output_cols = getattr(model, "get_output_cols", None)
            preferred   = None
            if callable(output_cols):
                cols = output_cols()
                preferred = cols[0] if cols else None

            pred_col = preferred or next(
                (c for c in pred_out.columns if "PREDICT" in c.upper() or "OUTPUT" in c.upper()),
                pred_out.columns[0],
            )
            predictions = np.asarray(pred_out[pred_col], dtype=np.float64).ravel()
        else:
            predictions = np.asarray(pred_out).ravel()

        missing_ctx  = [c for c in self.context_cols if c not in input_df.columns]
        if missing_ctx:
            raise ValueError(f"Missing context columns: {missing_ctx}")

        out = {c: input_df[c].values for c in self.context_cols}
        out[self.partition_col]  = partition_value
        out[self.prediction_col] = predictions
        return pd.DataFrame(out)


# %% [markdown]
# ## 4. Build Model Context and Partitioned Model
#
# Builds a `ModelContext` that holds all 16 sub-model instances (keyed by
# lower-cased group name), then wraps them into `PartitionedModel` so
# Snowflake can execute predictions partition-by-partition.

# %%
models_dict = {gn.lower(): info["model"] for gn, info in loaded_models.items()}
model_context    = custom_model.ModelContext(models=models_dict)
partitioned_model = PartitionedModel(
    model_context=model_context,
    feature_cols=feature_cols,
    partition_col=STATS_NTILE_GROUP_COL,
    context_cols=CONTEXT_COLS,
    prediction_col=PREDICTION_COL,
)
print(f"Partitioned model created with {len(models_dict)} sub-models")

# %% [markdown]
# ## 5. Prepare Sample Input
#
# Creates a small Snowpark-derived pandas sample with the exact feature
# columns expected by the native sub-models. This sample is embedded in
# the registry entry to support consistent metadata and routing.

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
    # Only keep rows for groups that actually have native sub-models loaded.
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

# Materialize the small sample to pandas for `registry.log_model(sample_input_data=...)`.
sample_input = sample_input_sp.to_pandas()
for col in ("WEEK", "BRAND_PRES_RET", "PROD_KEY"):
    # Replace potential NULLs in the registry sample to keep the model
    # input schema stable (and avoid runtime surprises in the function).
    sample_input[col] = sample_input[col].fillna("UNKNOWN" if col != "WEEK" else "000000")
for col in ("CUSTOMER_ID", STATS_NTILE_GROUP_COL, "WEEK", "BRAND_PRES_RET", "PROD_KEY"):
    sample_input[col] = sample_input[col].astype(str)
print(f"Sample input: {len(sample_input)} rows  |  {len(feat_cols_for_sample)} features")

# %% [markdown]
# ## 6. Register Partitioned Model
#
# Registers the combined partitioned model in the Model Registry, using
# the sample input to bake in required metadata and routing logic.

# %%
print(f"Registering {MODEL_NAME} v_{VERSION_DATE} ...")

# Collect the concrete version name for each sub-model (resolves the PRODUCTION alias)
import json as _json
submodel_versions = {}
for gn, info in loaded_models.items():
    mv_obj = info["model_version"]
    ver_name = getattr(mv_obj, "version_name", getattr(mv_obj, "name", str(mv_obj)))
    submodel_versions[gn] = ver_name

# Build the metrics dict: summary fields + one entry per sub-model version
# (used for filtering and inspection in the Model Registry UI).
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

# Persist the partitioned model as a TABLE_FUNCTION in the Model Registry.
mv = registry.log_model(
    partitioned_model,
    model_name=MODEL_NAME,
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
print(f"Registered: {MODEL_NAME} v_{VERSION_DATE}")
print("Sub-model versions baked in:")
for gn, ver in sorted(submodel_versions.items()):
    print(f"  {gn}: {ver}")

model_fqn = f"{DATABASE}.{MODELS_SCHEMA}.{MODEL_NAME}"
try:
    session.sql(f"ALTER MODEL {model_fqn} VERSION PRODUCTION UNSET ALIAS").collect()
    # Remove the previous PRODUCTION pointer so we can safely set the new one.
    print("Previous PRODUCTION alias removed")
except Exception:
    pass
session.sql(f"ALTER MODEL {model_fqn} VERSION v_{VERSION_DATE} SET ALIAS=PRODUCTION").collect()
print("PRODUCTION alias assigned to new version")

# %% [markdown]
# ## 7. Verify Registration
#
# Performs a lightweight registry check to confirm the new partitioned
# model version exists after registration.

# %%
result = session.sql(f"""
    SHOW MODELS LIKE '{MODEL_NAME}' IN SCHEMA {DATABASE}.{MODELS_SCHEMA}
""").collect()

if result:
    versions = session.sql(f"""
        SHOW VERSIONS IN MODEL {DATABASE}.{MODELS_SCHEMA}.{MODEL_NAME}
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
print(f"  Name:     {MODEL_NAME}")
print(f"  Version:  v_{VERSION_DATE}")
print(f"  Alias:    PRODUCTION")
print(f"  Groups:   {len(loaded_models)} ({', '.join(sorted(loaded_models.keys())[:4])}...)")
print(f"  Features: {len(feature_cols)}")
print("\nNext: 06a_setup_baselines.py")
