# %% [markdown]
# # Hyperparameter Search — Random Search (LGBM / XGB per group)
#
# Runs HPO per STATS_NTILE_GROUP in a sequential loop using Snowflake ML
# `tune.search.RandomSearch`.  Results are stored in ML Experiments (primary)
# or in HYPERPARAMETER_RESULTS (fallback).

# %% [markdown]
# ## 1. Setup

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.ml.modeling.tune import Tuner, TunerConfig, get_tuner_context, randint, uniform
from snowflake.ml.modeling.tune.search import RandomSearch
from snowflake.ml.data.data_connector import DataConnector
from snowflake.ml.experiment import ExperimentTracking
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
TRAIN_TABLE_CLEANED         = f"{DATABASE}.{STORAGE_SCHEMA}.TRAIN_DATASET_CLEANED"
FEATURES_TABLE              = f"{DATABASE}.{FEATURES_SCHEMA}.UNI_BOX_FEATURES"
HYPERPARAMETER_RESULTS_TABLE = f"{DATABASE}.{MODELS_SCHEMA}.HYPERPARAMETER_RESULTS"

TARGET_COLUMN         = "UNI_BOX_WEEK"
STATS_NTILE_GROUP_COL = "STATS_NTILE_GROUP"
EXPERIMENT_DATE       = datetime.now().strftime("%Y%m%d")
EXPERIMENT_NAME       = f"hyperparameter_search_regression_{EXPERIMENT_DATE}"

# Metadata columns excluded from the feature set
EXCLUDED_COLS = [
    "CUSTOMER_ID", "BRAND_PRES_RET", "PROD_KEY",
    "WEEK", STATS_NTILE_GROUP_COL,
]

# HPO settings
NUM_TRIALS           = 15
MAX_CONCURRENT_TRIALS = 4
SAMPLE_RATE_PER_GROUP = 0.2

# Cluster scaling
CLUSTER_SIZE_HPO     = 5
CLUSTER_SIZE_MIN_HPO = 2
CLUSTER_SIZE_DOWN    = 1

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
# ### 1B. Search spaces

# %%
SEARCH_SPACES = {
    "XGBRegressor": {
        "n_estimators":    randint(50, 300),
        "max_depth":       randint(3, 10),
        "learning_rate":   uniform(0.01, 0.3),
        "subsample":       uniform(0.6, 1.0),
        "colsample_bytree": uniform(0.6, 1.0),
        "min_child_weight": randint(1, 7),
        "gamma":           uniform(0, 0.5),
        "reg_alpha":       uniform(0, 1),
        "reg_lambda":      uniform(0, 1),
    },
    "LGBMRegressor": {
        "n_estimators":      randint(50, 300),
        "max_depth":         randint(3, 10),
        "learning_rate":     uniform(0.01, 0.3),
        "num_leaves":        randint(20, 150),
        "subsample":         uniform(0.6, 1.0),
        "colsample_bytree":  uniform(0.6, 1.0),
        "reg_alpha":         uniform(0, 1),
        "reg_lambda":        uniform(0, 1),
        "min_child_samples": randint(5, 50),
    },
}

# %% [markdown]
# ## 2. Load Groups and Training Data

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
print(f"Groups found: {len(groups_list)}")
if len(groups_list) != 16:
    print(f"WARNING: expected 16, found {len(groups_list)}")

# Load features table → join target; fallback to cleaned table
try:
    features_df = session.table(FEATURES_TABLE)
    target_df = session.table(TRAIN_TABLE_CLEANED).select(
        "CUSTOMER_ID", "BRAND_PRES_RET", "PROD_KEY", "WEEK",
        TARGET_COLUMN, STATS_NTILE_GROUP_COL,
    )
    train_df = features_df.join(
        target_df, on=["CUSTOMER_ID", "BRAND_PRES_RET", "PROD_KEY", "WEEK"], how="inner"
    )
    print(f"Training data: {train_df.count():,} rows (from features table)")
except Exception as e:
    print(f"Features table unavailable ({str(e)[:80]}), falling back to cleaned table")
    train_df = session.table(TRAIN_TABLE_CLEANED)
    print(f"Training data: {train_df.count():,} rows (from cleaned table)")

# %% [markdown]
# ## 3. Scale Cluster and Ray Dashboard

# %%
try:
    from snowflake.ml.runtime_cluster import scale_cluster
    scale_cluster(expected_cluster_size=CLUSTER_SIZE_HPO,
                  options={"block_until_min_cluster_size": CLUSTER_SIZE_MIN_HPO})
    print(f"Cluster scaled to {CLUSTER_SIZE_HPO} nodes")
except Exception as e:
    print(f"scale_cluster: {str(e)[:150]}")

try:
    from snowflake.ml.runtime_cluster import get_ray_dashboard_url
    print(f"Ray Dashboard: {get_ray_dashboard_url()}")
except Exception as e:
    print(f"Ray Dashboard: {str(e)[:100]}")

# %% [markdown]
# ## 4. Initialise ML Experiments

# %%
try:
    exp_tracking = ExperimentTracking(session)
    exp_tracking.set_experiment(EXPERIMENT_NAME)
    experiments_available = True
    print(f"Experiment: {EXPERIMENT_NAME}")
except Exception as e:
    print(f"ML Experiments not available ({str(e)[:100]}), using table fallback")
    exp_tracking = None
    experiments_available = False

if not experiments_available:
    session.sql(f"""
        CREATE TABLE IF NOT EXISTS {HYPERPARAMETER_RESULTS_TABLE} (
            search_id   VARCHAR,
            group_name  VARCHAR,
            algorithm   VARCHAR,
            best_params VARIANT,
            best_cv_rmse FLOAT,
            val_rmse    FLOAT,
            val_mae     FLOAT,
            n_iter      INTEGER,
            sample_size INTEGER,
            created_at  TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """).collect()
    print(f"Fallback table ready: {HYPERPARAMETER_RESULTS_TABLE}")

# %% [markdown]
# ## 5. Helper Functions

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


def temporal_train_val_split(snowpark_df, feat_cols, target_col, test_fraction=0.2):
    """Split a Snowpark DataFrame into temporal train and validation sets.

    Data is pulled from Snowflake ordered by ``WEEK`` (ascending) so that the
    validation set always corresponds to the most recent time periods.  This
    avoids data leakage that would occur with a random shuffle.

    Parameters
    ----------
    snowpark_df : snowflake.snowpark.DataFrame
        Source DataFrame containing at minimum the columns in ``feat_cols``,
        ``target_col``, and ideally a ``WEEK`` column for ordering.
    feat_cols : list[str]
        Feature column names to include in X.
    target_col : str
        Name of the target column.
    test_fraction : float, optional
        Fraction of the most-recent records to reserve for validation.
        Default is 0.2 (last 20% of weeks = test set).

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series]
        ``(X_train, X_val, y_train, y_val)`` — all pandas objects, WEEK
        column excluded from X.
    """
    from snowflake.snowpark import functions as F
    import numpy as np

    week_col = next((c for c in snowpark_df.columns if c.upper() == "WEEK"), None)

    select_exprs = [
        *[F.coalesce(F.col(c), F.lit(0)).alias(c) for c in feat_cols],
        F.coalesce(F.col(target_col), F.lit(0)).alias(target_col),
    ]
    if week_col:
        select_exprs.append(F.col(week_col).alias("_WEEK_SORT"))

    df = snowpark_df.select(*select_exprs).to_pandas()

    if "_WEEK_SORT" in df.columns:
        df = df.sort_values("_WEEK_SORT", ascending=True, ignore_index=True)
        df = df.drop(columns=["_WEEK_SORT"])

    split_idx   = int(len(df) * (1.0 - test_fraction))
    train_df_pd = df.iloc[:split_idx].reset_index(drop=True)
    val_df_pd   = df.iloc[split_idx:].reset_index(drop=True)

    return (
        train_df_pd[feat_cols],
        val_df_pd[feat_cols],
        train_df_pd[target_col],
        val_df_pd[target_col],
    )


stats_ntile_col = STATS_NTILE_GROUP_COL
_sample = train_df.filter(train_df[stats_ntile_col] == groups_list[0]).limit(1).to_pandas()
feature_cols = _get_feature_cols_numeric(_sample, EXCLUDED_COLS)
print(f"Feature columns detected: {len(feature_cols)}")

all_results = {}

# %% [markdown]
# ## 6. Training Function for Tuner

# %%
def create_train_func_for_tuner(feature_cols, model_type, target_col):
    """Create a training closure for the Snowflake ML Tuner.

    Parameters
    ----------
    feature_cols : list[str]
        Feature column names.
    model_type : str
        ``"XGBRegressor"`` or ``"LGBMRegressor"``.
    target_col : str
        Actual target column name in the dataset.

    Returns
    -------
    Callable
        Zero-argument training function for ``Tuner``.
    """
    def train_func():
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import numpy as np

        tuner_context = get_tuner_context()
        params = tuner_context.get_hyper_params()
        dm = tuner_context.get_dataset_map()

        train_pd      = dm["train"].to_pandas()
        test_pd       = dm["test"].to_pandas()
        train_dataset = train_pd[feature_cols + [target_col]].fillna(0)
        test_features = test_pd[feature_cols].fillna(0)
        y_val         = test_pd[target_col].fillna(0).values

        model_params = {**params, "random_state": 42}

        if model_type == "LGBMRegressor":
            from snowflake.ml.modeling.lightgbm import LGBMRegressor
            model_params.update(n_jobs=-1, verbosity=-1)
            model = LGBMRegressor(input_cols=feature_cols, label_cols=[target_col], **model_params)
        else:
            from snowflake.ml.modeling.xgboost import XGBRegressor
            model_params.update(n_jobs=-1, objective="reg:squarederror", eval_metric="rmse")
            model = XGBRegressor(input_cols=feature_cols, label_cols=[target_col], **model_params)

        model.fit(train_dataset)
        pred_df     = model.predict(test_features)
        pred_pandas = pred_df.to_pandas() if hasattr(pred_df, "to_pandas") else pred_df
        y_pred      = np.asarray(pred_pandas[model.get_output_cols()[0]])

        val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        val_mae  = float(mean_absolute_error(y_val, y_pred))
        denom    = np.sum(np.abs(y_val))
        val_wape = float(np.sum(np.abs(y_val - y_pred)) / denom) if denom > 0 else 0.0
        mask     = np.abs(y_val) > 1e-8
        val_mape = float((np.abs(y_val[mask] - y_pred[mask]) / np.abs(y_val[mask])).mean() * 100) if mask.any() else 0.0

        tuner_context.report(
            metrics={"rmse": val_rmse, "mae": val_mae, "wape": val_wape, "mape": val_mape},
            model=model,
        )

    return train_func


# %% [markdown]
# ## 7. Search Function (per group)

# %%
def run_hyperparameter_search_for_one_group(group_name, group_snowpark_df):
    """Run Random Search HPO for one segment group.

    Parameters
    ----------
    group_name : str
        STATS_NTILE_GROUP value.
    group_snowpark_df : snowflake.snowpark.DataFrame
        Data filtered to this group.

    Returns
    -------
    dict or None
        ``{best_params, best_rmse, val_rmse, val_mae}`` on success, ``None`` on skip/error.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from snowflake.snowpark import functions as F
    import numpy as np

    print(f"\n[{group_name}]")

    group_count = group_snowpark_df.count()
    if group_count < 50:
        print(f"  Skipping: only {group_count} records (< 50)")
        return None

    if SAMPLE_RATE_PER_GROUP < 1.0:
        sampled_df = group_snowpark_df.sample(frac=SAMPLE_RATE_PER_GROUP)
        sampled_count = sampled_df.count()
    else:
        sampled_df    = group_snowpark_df
        sampled_count = group_count
    print(f"  Records: {sampled_count:,} ({SAMPLE_RATE_PER_GROUP*100:.0f}% of {group_count:,})")

    sample_row    = sampled_df.limit(1).to_pandas()
    target_col    = TARGET_COLUMN
    feat_cols     = _get_feature_cols_numeric(sample_row, EXCLUDED_COLS)

    if sampled_count < 20:
        print("  Skipping: not enough data for train/val split")
        return None

    # Temporal split: train = oldest 80% of weeks, val = most recent 20%
    X_train, X_val, y_train, y_val = temporal_train_val_split(
        sampled_df, feat_cols, target_col, test_fraction=0.2
    )
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  (temporal split — test = most recent 20% of weeks)")

    train_data = X_train.copy()
    train_data[target_col] = np.asarray(y_train)
    val_data = X_val.copy()
    val_data[target_col] = np.asarray(y_val)

    train_dc = DataConnector.from_dataframe(session.create_dataframe(train_data))
    val_dc   = DataConnector.from_dataframe(session.create_dataframe(val_data))

    model_type  = GROUP_MODEL.get(group_name, _DEFAULT_MODEL)
    train_func  = create_train_func_for_tuner(feat_cols, model_type, target_col)
    tuner_config = TunerConfig(
        metric="rmse", mode="min",
        search_alg=RandomSearch(),
        num_trials=NUM_TRIALS,
        max_concurrent_trials=MAX_CONCURRENT_TRIALS,
    )
    print(f"  Model: {model_type}  Trials: {NUM_TRIALS}")

    try:
        t0      = time.time()
        tuner   = Tuner(train_func, SEARCH_SPACES[model_type], tuner_config)
        results = tuner.run(dataset_map={"train": train_dc, "test": val_dc})

        best_result = results.best_result
        config_cols = [c for c in best_result.columns if str(c).startswith("config/")]
        best_params = {
            str(c).replace("config/", ""): (
                v.item() if hasattr(v := best_result[c].iloc[0], "item") else v
            )
            for c in config_cols
        }
        best_rmse  = float(best_result["rmse"].iloc[0])
        best_model = results.best_model

        pred_pd  = best_model.predict(X_val)
        pred_pd  = pred_pd.to_pandas() if hasattr(pred_pd, "to_pandas") else pred_pd
        y_pred   = np.asarray(pred_pd[best_model.get_output_cols()[0]])
        val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        val_mae  = float(mean_absolute_error(y_val, y_pred))
        print(f"  Done in {time.time()-t0:.0f}s  best_rmse={best_rmse:.4f}  val_rmse={val_rmse:.4f}")

        all_results[group_name] = dict(
            best_params=best_params, best_cv_rmse=best_rmse,
            val_rmse=val_rmse, val_mae=val_mae,
            sample_size=sampled_count, algorithm=model_type,
        )

        search_id  = f"tune_{group_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        saved_to_experiments = False

        if experiments_available:
            try:
                run_name = f"best_{group_name}_{datetime.now().strftime('%H%M%S')}"
                with exp_tracking.start_run(run_name):
                    exp_tracking.log_params(best_params)
                    exp_tracking.log_metrics({
                        "best_rmse": best_rmse, "val_rmse": val_rmse,
                        "val_mae": val_mae, "sample_size": sampled_count,
                        "num_trials": NUM_TRIALS,
                    })
                    exp_tracking.log_param("group_name", group_name)
                    exp_tracking.log_param("search_id",  search_id)
                    exp_tracking.log_param("algorithm",  model_type)
                saved_to_experiments = True
                print(f"  Logged to ML Experiments (run: {run_name})")
            except Exception as e:
                print(f"  Experiments error ({str(e)[:80]}), saving to table")

        if not saved_to_experiments:
            params_json    = json.dumps({k: float(v) if isinstance(v, (np.integer, np.floating)) else v for k, v in best_params.items()})
            params_escaped = params_json.replace("'", "''")
            session.sql(f"""
                INSERT INTO {HYPERPARAMETER_RESULTS_TABLE}
                (search_id, group_name, algorithm, best_params, best_cv_rmse, val_rmse, val_mae, n_iter, sample_size)
                VALUES (
                    '{search_id}', '{group_name}', '{model_type}',
                    PARSE_JSON('{params_escaped}'),
                    {best_rmse:.6f}, {val_rmse:.6f}, {val_mae:.6f},
                    {NUM_TRIALS}, {sampled_count}
                )
            """).collect()
            print(f"  Saved to table: {HYPERPARAMETER_RESULTS_TABLE}")

        return dict(best_params=best_params, best_rmse=best_rmse, val_rmse=val_rmse, val_mae=val_mae)

    except Exception as e:
        import traceback
        print(f"  Error: {str(e)[:200]}")
        print(f"  {traceback.format_exc()[:300]}")
        return None


# %% [markdown]
# ## 8. Execute Search Loop

# %%
start_time   = time.time()
group_results = {}

for idx, group_name in enumerate(groups_list, 1):
    print(f"\n[{idx}/{len(groups_list)}] {group_name}")
    group_df = train_df.filter(train_df[stats_ntile_col] == group_name)
    group_results[group_name] = run_hyperparameter_search_for_one_group(group_name, group_df)

elapsed_min   = (time.time() - start_time) / 60
successful    = sum(1 for r in group_results.values() if r is not None)
print(f"\nCompleted {successful}/{len(groups_list)} groups in {elapsed_min:.1f} min")

# %% [markdown]
# ## 9. Summary

# %%
if experiments_available:
    print(f"Results in ML Experiments: {EXPERIMENT_NAME}")
    for gn, r in list(all_results.items())[:5]:
        print(f"  {gn}: val_rmse={r['val_rmse']:.4f}  val_mae={r['val_mae']:.4f}")
    if len(all_results) > 5:
        print(f"  ... and {len(all_results)-5} more groups")
else:
    session.sql(f"""
        SELECT group_name, best_cv_rmse, val_rmse, val_mae, sample_size, created_at
        FROM {HYPERPARAMETER_RESULTS_TABLE}
        WHERE created_at >= DATEADD(HOUR, -1, CURRENT_TIMESTAMP())
        ORDER BY group_name
    """).show()

if all_results:
    rmses = [r["val_rmse"] for r in all_results.values()]
    print(f"Validation RMSE — avg: {np.mean(rmses):.4f}  min: {min(rmses):.4f}  max: {max(rmses):.4f}")

# %% [markdown]
# ## 10. Scale Cluster Down

# %%
try:
    from snowflake.ml.runtime_cluster import scale_cluster
    scale_cluster(expected_cluster_size=CLUSTER_SIZE_DOWN)
    print(f"Cluster scaled down to {CLUSTER_SIZE_DOWN} node")
except Exception as e:
    print(f"scale_cluster down: {str(e)[:120]}")

print(f"\nHyperparameter search complete — {successful}/{len(groups_list)} groups in {elapsed_min:.1f} min")
print("Next: 04_many_model_training.py")
