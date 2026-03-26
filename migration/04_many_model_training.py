# %% [markdown]
# # MMT: 16 Models (LGBM / XGB per stats_ntile_group)
#
# Loads per-group hyperparameters from ML Experiments (or the
# fallback table), trains one model per group using ManyModelTraining,
# and registers each model in the Snowflake Model Registry.

# %% [markdown]
# ## 1. Setup

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F
from snowflake.ml.modeling.distributors.many_model import ManyModelTraining
from snowflake.ml.registry import Registry
from snowflake.ml.experiment import ExperimentTracking
from snowflake.ml.model import task
from datetime import datetime
import numpy as np
import json
import time

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

# Input tables
TRAIN_TABLE_CLEANED         = f"{DATABASE}.{FEATURES_SCHEMA}.{FEATURE_STORE_NAME}__TRAIN"
FEATURES_TABLE              = f"{DATABASE}.{FEATURES_SCHEMA}.{FEATURE_STORE_NAME}"
HYPERPARAMETER_RESULTS_TABLE = f"{DATABASE}.{MODELS_SCHEMA}.HPO_{MODEL_NAME}"
MMT_STAGE                   = f"{DATABASE}.{MODELS_SCHEMA}.MMT_MODELS"

TARGET_COLUMN         = "UNI_BOX_WEEK"
STATS_NTILE_GROUP_COL = "STATS_NTILE_GROUP"
VERSION_DATE          = datetime.now().strftime("%Y%m%d_%H%M")
# For testing purposes, decoment this line and comment the next one (use the experiment date of the previous run)
# EXPERIMENT_DATE = "20260318"
EXPERIMENT_DATE       = datetime.now().strftime("%Y%m%d")
EXPERIMENT_NAME       = f"EXP_{MODEL_NAME}_BAYESIAN_{EXPERIMENT_DATE}"

# Metadata columns excluded from the feature set
EXCLUDED_COLS = [
    "CUSTOMER_ID", "BRAND_PRES_RET", "PROD_KEY",
    "WEEK", STATS_NTILE_GROUP_COL,
]

# Cluster scaling
CLUSTER_SIZE_MMT     = 5
CLUSTER_SIZE_MIN_MMT = 2
CLUSTER_SIZE_DOWN    = 1

# Optional: set to a fraction (e.g. 0.1) to test with a data subset; None = full dataset
MMT_SAMPLE_FRACTION = None

# Group → algorithm mapping
GROUP_MODEL = {
    "group_stat_0_1": "LGBMRegressor", "group_stat_0_2": "LGBMRegressor",
    "group_stat_0_3": "LGBMRegressor", "group_stat_0_4": "LGBMRegressor",
    "group_stat_1_1": "LGBMRegressor", "group_stat_1_2": "LGBMRegressor",
    "group_stat_1_3": "XGBRegressor",  "group_stat_1_4": "XGBRegressor",
    "group_stat_2_1": "LGBMRegressor", "group_stat_2_2": "LGBMRegressor",
    "group_stat_2_3": "XGBRegressor",  "group_stat_2_4": "XGBRegressor",
    "group_stat_3_1": "LGBMRegressor", "group_stat_3_2": "LGBMRegressor",
    "group_stat_3_3": "LGBMRegressor", "group_stat_3_4": "XGBRegressor",
}
_DEFAULT_MODEL = "XGBRegressor"

session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()
print(f"Session: {session.get_current_database()}.{session.get_current_schema()}")

# %% [markdown]
# ## 2. Registry and Stage
#
# Prepares the Model Registry connection and creates the stage where the
# ManyModelTraining artifacts will be written during training.

# %%
session.sql(f"CREATE STAGE IF NOT EXISTS {MMT_STAGE}").collect()
registry = Registry(session=session, database_name=DATABASE, schema_name=MODELS_SCHEMA)
print(f"Registry ready: {DATABASE}.{MODELS_SCHEMA}")

# %% [markdown]
# ## 3. Load Hyperparameters (Experiments → Table fallback)
#
# Loads per-group best hyperparameters from ML Experiments for `EXPERIMENT_DATE`.
# If Experiments are not available (or incomplete), it falls back to the
# persisted results table and uses the most recent entry per group.

# %%
hyperparams_by_group = {}
expected_groups = [
    row[STATS_NTILE_GROUP_COL]
    for row in session.sql(f"""
        SELECT DISTINCT {STATS_NTILE_GROUP_COL}
        FROM {TRAIN_TABLE_CLEANED}
        WHERE {STATS_NTILE_GROUP_COL} IS NOT NULL
        ORDER BY {STATS_NTILE_GROUP_COL}
    """).collect()
]

# Try experiment for EXPERIMENT_DATE
experiments_loaded = False
try:
    exp_tracking = ExperimentTracking(session)
    exp_tracking.set_experiment(EXPERIMENT_NAME)

    runs_df = session.sql(f"SHOW RUNS IN EXPERIMENT {EXPERIMENT_NAME}").collect()
    if runs_df:
        runs_by_group = {}
        for run in runs_df:
            rn = run["name"]
            try:
                params  = {p["name"]: p["value"] for p in session.sql(f"SHOW RUN PARAMETERS IN EXPERIMENT {EXPERIMENT_NAME} RUN {rn}").collect()}
                metrics = {m["name"]: float(m["value"]) for m in session.sql(f"SHOW RUN METRICS IN EXPERIMENT {EXPERIMENT_NAME} RUN {rn}").collect()}
                gn      = params.get("group_name")
                rmse    = metrics.get("val_rmse")
                if not gn or rmse is None:
                    continue
                hp = {k: v for k, v in params.items() if k not in ("group_name", "search_id", "algorithm")}
                if gn not in runs_by_group or rmse < runs_by_group[gn]["val_rmse"]:
                    runs_by_group[gn] = {"params": hp, "val_rmse": rmse,
                                         "search_id": params.get("search_id", rn),
                                         "algorithm": params.get("algorithm", _DEFAULT_MODEL)}
            except Exception:
                continue

        if runs_by_group:
            hyperparams_by_group = {
                gn: {"params": info["params"], "val_rmse": info["val_rmse"],
                     "search_id": info["search_id"], "algorithm": info["algorithm"]}
                for gn, info in runs_by_group.items()
            }
            experiments_loaded = True
            print(f"Loaded {len(hyperparams_by_group)} groups from ML Experiments ({EXPERIMENT_NAME})")
except Exception as e:
    print(f"Experiments not available for {EXPERIMENT_NAME} ({str(e)[:80]})")

# Table fallback
if not experiments_loaded or len(hyperparams_by_group) < len(expected_groups):
    try:
        rows = session.sql(f"""
            WITH latest AS (
                SELECT group_name, algorithm, best_params, val_rmse, search_id,
                       ROW_NUMBER() OVER (PARTITION BY group_name ORDER BY created_at DESC) AS rn
                FROM {HYPERPARAMETER_RESULTS_TABLE}
                WHERE group_name IS NOT NULL
            )
            SELECT group_name, algorithm, best_params, val_rmse, search_id
            FROM latest WHERE rn = 1
        """).collect()
        for r in rows:
            gn = r["GROUP_NAME"]
            if gn not in hyperparams_by_group:
                bp = json.loads(r["BEST_PARAMS"]) if isinstance(r["BEST_PARAMS"], str) else r["BEST_PARAMS"]
                hyperparams_by_group[gn] = {
                    "params": bp, "val_rmse": r["VAL_RMSE"],
                    "search_id": r["SEARCH_ID"],
                    "algorithm": r.get("ALGORITHM") or GROUP_MODEL.get(gn, _DEFAULT_MODEL),
                }
        print(f"Total hyperparameter groups loaded: {len(hyperparams_by_group)}")
    except Exception as e:
        print(f"Table fallback error: {str(e)[:200]}")

if not hyperparams_by_group:
    raise ValueError("No hyperparameter results found — run 03_hyperparameter_search.py first")

missing_groups = set(expected_groups) - set(hyperparams_by_group)
if missing_groups:
    print(f"WARNING: {len(missing_groups)} groups will use default hyperparameters: {sorted(missing_groups)}")

# %% [markdown]
# ### 3B. Default hyperparameters

# %%
DEFAULT_PARAMS_BY_MODEL = {
    "XGBRegressor": {
        "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1,
        "gamma": 0, "reg_alpha": 0, "reg_lambda": 1,
    },
    "LGBMRegressor": {
        "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
        "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0, "reg_lambda": 1, "min_child_samples": 20,
    },
}

# %% [markdown]
# ## 4. Load Training Data
#
# Loads the engineered feature table and joins it with the labels/segment
# identifiers from the cleaned training table, preparing the input dataset
# for ManyModelTraining.

# %%
try:
    features_df = session.table(FEATURES_TABLE)
    target_df   = session.table(TRAIN_TABLE_CLEANED).select(
        "CUSTOMER_ID", "BRAND_PRES_RET", "PROD_KEY", "WEEK",
        TARGET_COLUMN, STATS_NTILE_GROUP_COL,
    )
    training_df = features_df.join(
        target_df, on=["CUSTOMER_ID", "BRAND_PRES_RET", "PROD_KEY", "WEEK"], how="inner"
    )
    print(f"Training data: {training_df.count():,} rows (from features table)")
except Exception as e:
    print(f"Features table unavailable ({str(e)[:80]}), falling back to cleaned table")
    training_df = session.table(TRAIN_TABLE_CLEANED)
    print(f"Training data: {training_df.count():,} rows (from cleaned table)")

if MMT_SAMPLE_FRACTION is not None and 0 < MMT_SAMPLE_FRACTION < 1:
    training_df = training_df.sample(frac=MMT_SAMPLE_FRACTION)
    print(f"Sampled to {MMT_SAMPLE_FRACTION*100:.0f}%: {training_df.count():,} rows")

print("Rows per group:")
training_df.group_by(STATS_NTILE_GROUP_COL).count().sort(STATS_NTILE_GROUP_COL).show(n=20)

# %% [markdown]
# ## 5. Helper Functions
#
# Helper utilities used by the per-partition training function:
# - numeric feature column selection
# - temporal train/test split defined per `STATS_NTILE_GROUP`
# - the `train_segment_model` closure executed by ManyModelTraining workers.

# %%
def _get_feature_cols_numeric(df, excluded_cols):
    """Return numeric feature columns, excluding metadata/target.

    Parameters
    ----------
    df : pandas.DataFrame
        Schema-reference DataFrame (may be a single-row sample).
    excluded_cols : list[str]
        Column names to exclude (e.g., metadata, target).

    Returns
    -------
    list[str]
        Numeric feature column names.
    """
    excluded_set = set(excluded_cols)
    excluded_set.add(TARGET_COLUMN)
    return [
        col for col in df.columns
        if col not in excluded_set
        and getattr(df[col].dtype, "kind", "O") in "iufb"
    ]


def temporal_train_val_split_pandas(df, feat_cols, target_col, test_fraction=0.2):
    """Split a pandas DataFrame into temporal train and test sets (per group).

    This function implements a temporal holdout that is computed per `STATS_NTILE_GROUP`,
    mirroring the cutoff-week logic used in `01_data_validation_and_cleaning.py`.
    For each group, we compute:
    1) row counts per (`STATS_NTILE_GROUP`, `WEEK`),
    2) cumulative share ordered by `WEEK` ascending,
    3) `cutoff_week` as the first week where cumulative share reaches/exceeds
       ``1 - test_fraction``,
    4) `train` as weeks ``<= cutoff_week`` and `test` as weeks ``> cutoff_week``.

    Note: although `ManyModelTraining` already partitions by `STATS_NTILE_GROUP`,
    this per-group cutoff keeps the temporal definition consistent and avoids
    any accidental reliance on input ordering.

    Parameters
    ----------
    df : pandas.DataFrame
        Pandas DataFrame already fetched from Snowflake.
        Must contain `WEEK` and `STATS_NTILE_GROUP_COL` plus the columns in `feat_cols`
        and `target_col`.
    feat_cols : list[str]
        Feature column names to include in X.
    target_col : str
        Name of the target column.
    test_fraction : float, optional
        Fraction of the most-recent weeks reserved for test (per group).
        Default is 0.2.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series]
        ``(X_train, X_test, y_train, y_test)`` — the `WEEK` column is excluded
        from X via column selection using `feat_cols`.
    """
    import pandas as pd

    week_col = next((c for c in df.columns if c.upper() == "WEEK"), None)
    group_col = next(
        (c for c in df.columns if c.upper() == STATS_NTILE_GROUP_COL.upper()), None
    )

    # Fallback: if missing required columns, do a global temporal split.
    if not week_col or not group_col:
        sorted_df = df.sort_values(week_col, ascending=True, ignore_index=True) if week_col else df
        split_idx = int(len(sorted_df) * (1.0 - test_fraction))
        train_df_pd = sorted_df.iloc[:split_idx].reset_index(drop=True)
        test_df_pd = sorted_df.iloc[split_idx:].reset_index(drop=True)
        return (
            train_df_pd[feat_cols],
            test_df_pd[feat_cols],
            train_df_pd[target_col],
            test_df_pd[target_col],
        )

    threshold = 1.0 - float(test_fraction)
    train_parts = []
    test_parts = []

    for _, gdf in df.groupby(group_col):
        gdf_sorted = gdf.sort_values(week_col, ascending=True)

        weekly = (
            gdf_sorted.groupby(week_col)
            .size()
            .reset_index(name="weekly_rows")
            .sort_values(week_col)
        )

        weekly["running_total"] = weekly["weekly_rows"].cumsum()
        total_group_rows = float(weekly["weekly_rows"].sum())
        weekly["total_group_rows"] = total_group_rows
        # Use a scalar guard here; `weekly["total_group_rows"]` is a Series and can't be used in `if`.
        weekly["time_share"] = (weekly["running_total"] / total_group_rows) if total_group_rows > 0 else 0.0

        if (weekly["time_share"] >= threshold).any():
            # Cutoff is the first week where cumulative share reaches the desired threshold.
            cutoff_week = weekly.loc[weekly["time_share"] >= threshold, week_col].min()
        else:
            # Degenerate case: if the threshold is never reached (e.g., tiny groups),
            # fall back to putting all rows into train.
            cutoff_week = weekly[week_col].max()

        train_sub = gdf_sorted[gdf_sorted[week_col] <= cutoff_week].reset_index(drop=True)
        test_sub = gdf_sorted[gdf_sorted[week_col] > cutoff_week].reset_index(drop=True)
        train_parts.append(train_sub)
        test_parts.append(test_sub)

    train_df_pd = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0].copy()
    test_df_pd = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0].copy()

    return (
        train_df_pd[feat_cols],
        test_df_pd[feat_cols],
        train_df_pd[target_col],
        test_df_pd[target_col],
    )


# %% [markdown]
# ## 6. MMT Training Function
#
# Defines the function executed in parallel by `ManyModelTraining` for each
# partition (group). It builds the train/test matrices, trains the selected
# regressor using the group hyperparameters, evaluates metrics, and attaches
# them to the returned estimator for later logging/registration.
#
# ### Note on where models run (MMT execution model)
#
# A common concern is whether importing `snowflake.ml.modeling.*` estimators means
# we are instantiating a “distributed model inside a node”.
#
# - **`ManyModelTraining` is the distributed system**: it partitions the input dataset
#   (e.g., by `STATS_NTILE_GROUP`) and schedules **one partition per worker** in the
#   Snowflake-managed runtime.
# - Inside each worker, this function trains **a regular, single-partition estimator**
#   on a **pandas DataFrame** for that partition. There is no nested distributed training
#   happening inside the estimator.
# - The estimators in `snowflake.ml.modeling.*` are **runtime-compatible wrappers** designed
#   to work reliably inside Snowflake (dependencies, serialization, remote execution).
#   Using “native” libraries directly (e.g., `xgboost.XGBRegressor`) can work in some setups,
#   but may break packaging/serialization in the Snowflake runtime; hence the wrappers are
#   the recommended choice for MMT.

# %%
def train_segment_model(data_connector, context):
    """Train one segment model inside ManyModelTraining.

    Parameters
    ----------
    data_connector : snowflake.ml.data.DataConnector
        Provides ``to_pandas()`` access to the partition's data.
    context : ManyModelTrainingContext
        Exposes ``context.partition_id`` for the current partition.

    Returns
    -------
    snowflake.ml.modeling.BaseEstimator
        Fitted model with extra attributes ``rmse``, ``mae``, ``wape``,
        ``mape``, ``training_samples``, ``test_samples``, ``feature_cols``,
        ``hyperparameters``, and ``group_name``.
    """
    import pandas as pd
    from snowflake.ml.modeling.xgboost import XGBRegressor
    from snowflake.ml.modeling.lightgbm import LGBMRegressor
    from snowflake.ml.modeling.linear_model import SGDRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np

    segment = context.partition_id
    print(f"\n[{segment}]")

    # Each ManyModelTraining partition receives a single `STATS_NTILE_GROUP` slice.
    df         = data_connector.to_pandas()
    target_col = TARGET_COLUMN
    feat_cols  = _get_feature_cols_numeric(df, EXCLUDED_COLS)
    if len(feat_cols) < 5:
        excluded_set = set(EXCLUDED_COLS)
        excluded_set.add(TARGET_COLUMN)
        feat_cols = [c for c in df.columns if c not in excluded_set]

    _target_series = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
    print(f"  Data: {df.shape}  target range [{_target_series.min():.2f}, {_target_series.max():.2f}]")

    # Temporal split: train = oldest 80% of weeks, test = most recent 20%
    X_train, X_test, y_train, y_test = temporal_train_val_split_pandas(
        df, feat_cols, target_col, test_fraction=0.2
    )
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}  (temporal split — test = most recent 20% of weeks)")

    # Prepare numeric model matrices + labels.
    train_dataset = X_train.apply(pd.to_numeric, errors="coerce").astype(float)
    train_dataset[target_col] = np.asarray(y_train, dtype=np.float64)

    test_features = X_test.apply(pd.to_numeric, errors="coerce").astype(float)

    # Resolve algorithm and hyperparameters
    model_type = GROUP_MODEL.get(segment, _DEFAULT_MODEL)
    if segment in hyperparams_by_group:
        alg = hyperparams_by_group[segment].get("algorithm")
        if alg:
            model_type = alg
        raw_params  = hyperparams_by_group[segment]["params"]
        search_id   = hyperparams_by_group[segment]["search_id"]
        print(f"  Optimised params (search_id={search_id}, model={model_type})")
    else:
        raw_params = DEFAULT_PARAMS_BY_MODEL.get(model_type, DEFAULT_PARAMS_BY_MODEL["XGBRegressor"])
        print(f"  Default params (model={model_type})")

    INT_PARAMS_SET  = {"n_estimators", "max_depth", "num_leaves", "min_child_weight", "min_child_samples", "max_iter"}
    defaults        = DEFAULT_PARAMS_BY_MODEL.get(model_type, DEFAULT_PARAMS_BY_MODEL["XGBRegressor"])
    model_params    = {"random_state": 42}
    # Cast/tidy hyperparameters so model constructors accept the right native types.
    for k, v in raw_params.items():
        native = v.item() if hasattr(v, "item") else v
        try:
            if k in INT_PARAMS_SET:
                model_params[k] = int(float(native))
            else:
                model_params[k] = float(native) if isinstance(native, (int, float, np.integer, np.floating)) else native
        except (TypeError, ValueError):
            model_params[k] = defaults.get(k, native)

    MODEL_CLASSES = {"XGBRegressor": XGBRegressor, "LGBMRegressor": LGBMRegressor, "SGDRegressor": SGDRegressor}
    ModelClass = MODEL_CLASSES.get(model_type, XGBRegressor)
    if model_type == "XGBRegressor":
        model_params.update(n_jobs=-1, objective="reg:squarederror", eval_metric="rmse")
    elif model_type == "LGBMRegressor":
        model_params.update(n_jobs=-1, verbosity=-1)
    elif model_type == "SGDRegressor":
        model_params.setdefault("penalty", "l2")
        model_params.setdefault("learning_rate", "invscaling")

    model = ModelClass(input_cols=feat_cols, label_cols=[target_col], **model_params)
    model.fit(train_dataset)

    # Predict + compute evaluation metrics on the temporal holdout.
    pred_pd = model.predict(test_features)
    pred_pd = pred_pd.to_pandas() if hasattr(pred_pd, "to_pandas") else pred_pd
    y_pred  = np.asarray(pred_pd[model.get_output_cols()[0]])

    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae    = float(mean_absolute_error(y_test, y_pred))
    denom  = np.sum(np.abs(y_test))
    # WAPE is undefined when sum(|y|) == 0 (e.g., all actuals are 0).
    # Returning 0.0 would be misleading if predictions are non-zero, so we emit NaN.
    wape   = float(np.sum(np.abs(y_test - y_pred)) / denom) if denom > 0 else float("nan")
    mask   = np.abs(y_test) > 1e-8
    mape   = float((np.abs(y_test[mask] - y_pred[mask]) / np.abs(y_test[mask])).mean() * 100) if mask.any() else 0.0

    print(f"  RMSE={rmse:.2f}  MAE={mae:.2f}  WAPE={wape:.4f}  MAPE={mape:.2f}%")

    # Attach metrics and metadata to the returned estimator for downstream
    # registry logging in the main driver.
    model.rmse             = rmse
    model.mae              = mae
    model.wape             = wape
    model.mape             = mape
    model.training_samples = X_train.shape[0]
    model.test_samples     = X_test.shape[0]
    model.feature_cols     = feat_cols
    model.hyperparameters  = model_params
    model.group_name       = segment
    return model


# %% [markdown]
# ## 7. Scale Cluster, Ray Dashboard, Run MMT
#
# Scales the runtime cluster for ManyModelTraining, optionally prints the
# Ray dashboard URL, then launches one training job per partition
# (partitioned by `STATS_NTILE_GROUP`).

# %%
try:
    from snowflake.ml.runtime_cluster import scale_cluster
    scale_cluster(expected_cluster_size=CLUSTER_SIZE_MMT,
                  options={"block_until_min_cluster_size": CLUSTER_SIZE_MIN_MMT})
    print(f"Cluster scaled to {CLUSTER_SIZE_MMT} nodes")
except Exception as e:
    print(f"scale_cluster: {str(e)[:150]}")

try:
    from snowflake.ml.runtime_cluster import get_ray_dashboard_url
    print(f"Ray Dashboard: {get_ray_dashboard_url()}")
except Exception as e:
    print(f"Ray Dashboard: {str(e)[:100]}")

# %%
start_time   = time.time()
trainer      = ManyModelTraining(train_segment_model, MMT_STAGE)
# Launch distributed training for each `STATS_NTILE_GROUP` partition.
training_run = trainer.run(
    partition_by=STATS_NTILE_GROUP_COL,
    snowpark_dataframe=training_df,
    run_id=f"{MODEL_NAME.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
print(f"Run ID: {training_run.run_id}")

# %% [markdown]
# ## 8. Wait for MMT Completion
#
# Polls `training_run.partition_details` until all partitions finish (DONE)
# and blocks until the run reaches a terminal state.

# %%
MMT_CHECK_INTERVAL  = 30
elapsed = 0
consecutive_poll_errors = 0
POLL_ERROR_LIMIT = 5
all_finished = False

while True:
    time.sleep(MMT_CHECK_INTERVAL)
    elapsed += MMT_CHECK_INTERVAL
    try:
        details     = training_run.partition_details
        consecutive_poll_errors = 0
        total_count = len(details)
        done        = sum(1 for d in details.values() if d.status.name == "DONE")
        failed      = sum(1 for d in details.values() if d.status.name == "FAILED")
        print(f"  {elapsed}s — OK: {done}  FAILED: {failed}  pending: {total_count-done-failed}", end="\r")
        if done + failed == total_count:
            print(f"\nAll {total_count} partitions finished")
            all_finished = True
            break
    except Exception as e:
        print(f"\npartition_details error: {str(e)[:180]}")
        # Transient network/API errors shouldn't end the wait loop; retry on the
        # next polling cycle so we don't proceed to registration prematurely.
        consecutive_poll_errors += 1
        if consecutive_poll_errors >= POLL_ERROR_LIMIT:
            print(f"Too many consecutive polling errors ({POLL_ERROR_LIMIT}). Aborting wait.")
            break
        continue

# Safety guard: do not proceed to registration if we couldn't confirm completion.
if not all_finished:
    raise RuntimeError(
        "MMT did not reach a confirmed terminal state (DONE/FAILED for all partitions). "
        "Aborting to avoid registering incomplete results."
    )

# %% [markdown]
# ## 9. Partition Results
#
# Summarizes DONE vs FAILED partitions and prints key metrics (RMSE/MAE)
# when available.

# %%
try:
    partition_details = training_run.partition_details
except Exception as e:
    partition_details = {}
    print(f"Could not read partition_details: {str(e)[:200]}")

done_ids, failed_ids = [], []
for pid, details in partition_details.items():
    st = details.status.name
    if st == "DONE":
        done_ids.append(pid)
        try:
            m = training_run.get_model(pid)
            print(f"  {pid}: RMSE={m.rmse:.2f}  MAE={m.mae:.2f}  samples={m.training_samples:,}")
        except Exception as e:
            print(f"  {pid}: DONE but model load failed ({str(e)[:80]})")
    elif st == "FAILED":
        failed_ids.append(pid)
        print(f"  {pid}: FAILED")
    else:
        print(f"  {pid}: {st}")
print(f"\nSummary: {len(done_ids)} OK  {len(failed_ids)} FAILED  {len(partition_details)-len(done_ids)-len(failed_ids)} pending")

# %% [markdown]
# ## 10. Register Models
#
# For every successful partition, logs the trained model into the registry,
# attaches per-group metrics and hyperparameters, and sets the `PRODUCTION`
# alias to the newly trained version.

# %%
registered_models = {}
for pid, details in partition_details.items():
    if details.status.name != "DONE":
        continue
    try:
        model       = training_run.get_model(pid)
        model_name  = f"{MODEL_NAME.lower()}__{pid.lower()}"
        group_hp    = hyperparams_by_group.get(pid, {})
        group_alg   = group_hp.get("algorithm") or GROUP_MODEL.get(pid, _DEFAULT_MODEL)
        group_sid   = group_hp.get("search_id", "default")
        raw_hparams = group_hp.get("params", {})

        # Build registry metrics: core metrics + optional hyperparameters snapshot.
        metrics = {
            "rmse": float(model.rmse), "mae": float(model.mae),
            # Keep WAPE as NaN when undefined (sum(|y|)==0). Note: depending on the
            # backend/serialization, NaN may be stored as NULL; we prefer NaN over 0.0.
            "wape": float(model.wape),
            "mape": float(model.mape),
            "training_samples": int(model.training_samples),
            "test_samples": int(model.test_samples),
            "algorithm": group_alg, "group": pid,
            "hyperparameter_search_id": group_sid,
            "feature_table_name": FEATURES_TABLE,
        }
        if raw_hparams:
            metrics["hyperparameters"] = json.dumps(
                {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                 for k, v in raw_hparams.items()}
            )

        mv = registry.log_model(
            model,
            model_name=model_name,
            version_name=f"v_{VERSION_DATE}",
            comment=f"{group_alg} model for uni_box_week — Group: {pid}",
            metrics=metrics,
            task=task.Task.TABULAR_REGRESSION,
        )
        registered_models[pid] = {"model_name": model_name, "version": f"v_{VERSION_DATE}", "model_version": mv}
        print(f"  {pid}: {model_name} v_{VERSION_DATE}")

        # Set PRODUCTION alias
        model_fqn = f"{DATABASE}.{MODELS_SCHEMA}.{model_name}"
        try:
            session.sql(f"ALTER MODEL {model_fqn} VERSION PRODUCTION UNSET ALIAS").collect()
        except Exception:
            pass
        session.sql(f"ALTER MODEL {model_fqn} VERSION v_{VERSION_DATE} SET ALIAS=PRODUCTION").collect()
        print(f"    PRODUCTION alias set")
    except Exception as e:
        print(f"  Error registering {pid}: {str(e)[:200]}")

print(f"\n{len(registered_models)}/16 models registered")

# %% [markdown]
# ## 11. Verify PRODUCTION Alias
#
# Lightweight check to confirm the `PRODUCTION` alias points to the new
# registered version for each successfully trained group model.

# %%
for pid, info in registered_models.items():
    try:
        prod_v = registry.get_model(info["model_name"]).version("PRODUCTION")
        v_name = getattr(prod_v, "name", getattr(prod_v, "version_name", str(prod_v)))
        print(f"  {info['model_name']}: PRODUCTION -> {v_name}")
    except Exception as e:
        print(f"  {info['model_name']}: alias not found ({str(e)[:80]})")

# %% [markdown]
# ## 12. Scale Down and Summary

# %%
try:
    from snowflake.ml.runtime_cluster import scale_cluster
    scale_cluster(expected_cluster_size=CLUSTER_SIZE_DOWN)
    print(f"Cluster scaled down to {CLUSTER_SIZE_DOWN} node")
except Exception as e:
    print(f"scale_cluster down: {str(e)[:120]}")

elapsed_min = (time.time() - start_time) / 60
print(f"\nMMT complete — {len(registered_models)}/16 models  {elapsed_min:.1f} min")
print("Next: 05_create_partitioned_model.py")
