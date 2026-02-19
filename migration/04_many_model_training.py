# %% [markdown]
# # MMT: 16 Models (LGBM/XGB per stats_ntile_group)
# Hyperparameters per group from script 03 → train → register in Model Registry.

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F
from snowflake.ml.modeling.distributors.many_model import ManyModelTraining
from snowflake.ml.registry import Registry
from snowflake.ml.feature_store import FeatureStore
from snowflake.ml.experiment import ExperimentTracking
from snowflake.ml.model import task
import time
from datetime import datetime
import json

session = get_active_session()

# Configuration: Database, schemas, and tables
DATABASE = "BD_AA_DEV"
STORAGE_SCHEMA = "SC_STORAGE_BMX_PS"
FEATURES_SCHEMA = "SC_FEATURES_BMX"
MODELS_SCHEMA = "SC_MODELS_BMX"
TRAIN_TABLE_CLEANED = f"{DATABASE}.{STORAGE_SCHEMA}.TRAIN_DATASET_CLEANED"
FEATURES_TABLE = f"{DATABASE}.{FEATURES_SCHEMA}.UNI_BOX_FEATURES"
FEATURE_VERSIONS_TABLE = f"{DATABASE}.{FEATURES_SCHEMA}.FEATURE_VERSIONS"
HYPERPARAMETER_RESULTS_TABLE = f"{DATABASE}.{MODELS_SCHEMA}.HYPERPARAMETER_RESULTS"
MMT_STAGE = f"{DATABASE}.{MODELS_SCHEMA}.MMT_MODELS"
DEFAULT_WAREHOUSE = "WH_AA_DEV_DS_SQL"

# Column constants
TARGET_COLUMN = "UNI_BOX_WEEK"
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

# Cluster scaling configuration
CLUSTER_SIZE_MMT = 5
CLUSTER_SIZE_MIN_MMT = 2
CLUSTER_SIZE_DOWN = 1

session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()
print(f"✅ {session.get_current_database()}.{session.get_current_schema()}")

# Resolve active feature version (for model traceability)
feature_version_id = None
feature_snapshot_at = None
try:
    _version_rows = session.sql(
        f"""
        SELECT FEATURE_VERSION_ID, FEATURE_SNAPSHOT_AT, CREATED_AT
        FROM {FEATURE_VERSIONS_TABLE}
        WHERE FEATURE_TABLE_NAME = 'UNI_BOX_FEATURES'
          AND IS_ACTIVE = TRUE
        ORDER BY CREATED_AT DESC
        LIMIT 1
    """
    ).collect()
    if _version_rows:
        feature_version_id = _version_rows[0]["FEATURE_VERSION_ID"]
        feature_snapshot_at = _version_rows[0]["FEATURE_SNAPSHOT_AT"]
        print("\n📌 Active feature version for model training:")
        print(f"   FEATURE_VERSION_ID: {feature_version_id}")
        print(f"   FEATURE_SNAPSHOT_AT: {feature_snapshot_at}")
    else:
        print("\n⚠️  No active feature version found in FEATURE_VERSIONS; models will be registered without feature version metadata")
except Exception as e:
    print(f"\n⚠️  Could not resolve feature version for models: {str(e)[:200]}")

USE_CLEANED_TABLES = False
MMT_SAMPLE_FRACTION = None  # None = 100%

GROUP_MODEL = {
    "group_stat_0_1": "LGBMRegressor",
    "group_stat_0_2": "LGBMRegressor",
    "group_stat_0_3": "LGBMRegressor",
    "group_stat_0_4": "LGBMRegressor",
    "group_stat_1_1": "LGBMRegressor",
    "group_stat_1_2": "LGBMRegressor",
    "group_stat_1_3": "XGBRegressor",
    "group_stat_1_4": "XGBRegressor",
    "group_stat_2_1": "LGBMRegressor",
    "group_stat_2_2": "LGBMRegressor",
    "group_stat_2_3": "XGBRegressor",
    "group_stat_2_4": "XGBRegressor",
    "group_stat_3_1": "LGBMRegressor",
    "group_stat_3_2": "LGBMRegressor",
    "group_stat_3_3": "LGBMRegressor",
    "group_stat_3_4": "XGBRegressor",
}
_DEFAULT_MODEL = "XGBRegressor"

# %% [markdown]
# ## 1. Registry and Stage

# %%
session.sql(f"CREATE STAGE IF NOT EXISTS {MMT_STAGE}").collect()
registry = Registry(session=session, database_name=DATABASE, schema_name=MODELS_SCHEMA)
print("✅ Registry + stage ready")

# %% [markdown]
# ## 2. Hyperparameters per Group (Experiments or Table)

# %%
hyperparams_by_group = {}
experiments_loaded = False
all_groups_from_data = session.sql(
    f"""
    SELECT DISTINCT {STATS_NTILE_GROUP_COL}
    FROM {TRAIN_TABLE_CLEANED}
    WHERE {STATS_NTILE_GROUP_COL} IS NOT NULL
    ORDER BY {STATS_NTILE_GROUP_COL}
"""
).collect()

expected_groups = [row[STATS_NTILE_GROUP_COL] for row in all_groups_from_data]

print("\n🔬 Loading from ML Experiments...")
try:
    exp_tracking = ExperimentTracking(session)
    from datetime import datetime, timedelta

    today = datetime.now().strftime("%Y%m%d")
    experiment_name = f"hyperparameter_search_regression_{today}"

    try:
        exp_tracking.set_experiment(experiment_name)
        print(f"✅ Found experiment: {experiment_name}")

        # Get all runs from this experiment
        experiments_loaded = True
        print("   ✅ ML Experiments available - loading from experiments")
    except:
        # Try yesterday's experiment as fallback
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        experiment_name = f"hyperparameter_search_regression_{yesterday}"
        try:
            exp_tracking.set_experiment(experiment_name)
            print(f"✅ Found experiment: {experiment_name}")
            experiments_loaded = True
        except:
            print("   ⚠️  No recent experiment found, will use table fallback")
            experiments_loaded = False

    if experiments_loaded:
        try:
            print(f"   📋 Runs en experiment: {experiment_name}")

            runs_query = f"SHOW RUNS IN EXPERIMENT {experiment_name}"
            runs_df = session.sql(runs_query)
            runs_list = runs_df.collect()

            if len(runs_list) == 0:
                print("   ⚠️  No runs found in experiment, using table fallback")
                experiments_loaded = False
            else:
                print(f"   ✅ Found {len(runs_list)} runs in experiment")

                runs_by_group = {}

                for run in runs_list:
                    run_name = run["name"]

                    try:
                        # Get parameters for this run
                        params_query = f"SHOW RUN PARAMETERS IN EXPERIMENT {experiment_name} RUN {run_name}"
                        params_df = session.sql(params_query)
                        params_list = params_df.collect()

                        # Get metrics for this run
                        metrics_query = f"SHOW RUN METRICS IN EXPERIMENT {experiment_name} RUN {run_name}"
                        metrics_df = session.sql(metrics_query)
                        metrics_list = metrics_df.collect()

                        # Extract group_name and algorithm from parameters
                        group_name = None
                        search_id = None
                        algorithm = None
                        best_params = {}

                        for param in params_list:
                            param_name = param["name"]
                            param_value = param["value"]

                            if param_name == "group_name":
                                group_name = param_value
                            elif param_name == "search_id":
                                search_id = param_value
                            elif param_name == "algorithm":
                                algorithm = param_value
                            else:
                                best_params[param_name] = param_value

                        # Extract metrics
                        val_rmse = None
                        val_mae = None

                        for metric in metrics_list:
                            metric_name = metric["name"]
                            metric_value = metric["value"]

                            if metric_name == "val_rmse":
                                val_rmse = float(metric_value)
                            elif metric_name == "val_mae":
                                val_mae = float(metric_value)

                        # Only process runs that have a group_name
                        if group_name and val_rmse is not None:
                            alg = algorithm or GROUP_MODEL.get(
                                group_name, _DEFAULT_MODEL
                            )
                            if group_name not in runs_by_group:
                                runs_by_group[group_name] = {
                                    "run_name": run_name,
                                    "params": best_params,
                                    "val_rmse": val_rmse,
                                    "val_mae": val_mae,
                                    "search_id": search_id,
                                    "algorithm": alg,
                                }
                            else:
                                if val_rmse < runs_by_group[group_name]["val_rmse"]:
                                    runs_by_group[group_name] = {
                                        "run_name": run_name,
                                        "params": best_params,
                                        "val_rmse": val_rmse,
                                        "val_mae": val_mae,
                                        "search_id": search_id,
                                        "algorithm": alg,
                                    }

                    except Exception as run_error:
                        print(
                            f"   ⚠️  Error processing run {run_name}: {str(run_error)[:100]}"
                        )
                        continue

                # Step 3: Store results in hyperparams_by_group
                if len(runs_by_group) > 0:
                    print(f"   ✅ Loaded {len(runs_by_group)} groups from Experiments")

                    for group_name, run_info in runs_by_group.items():
                        hyperparams_by_group[group_name] = {
                            "params": run_info["params"],
                            "val_rmse": run_info["val_rmse"],
                            "search_id": run_info["search_id"] or f"exp_{group_name}",
                            "algorithm": run_info.get("algorithm", _DEFAULT_MODEL),
                        }

                        print(f"\n   {group_name}:")
                        print(
                            f"      Algorithm: {run_info.get('algorithm', _DEFAULT_MODEL)}"
                        )
                        print(f"      Val RMSE: {run_info['val_rmse']:.4f}")
                        if run_info["val_mae"]:
                            print(f"      Val MAE: {run_info['val_mae']:.4f}")
                        print(f"      Search ID: {run_info['search_id'] or 'N/A'}")
                        print(
                            f"      Source: ML Experiments (run: {run_info['run_name']})"
                        )

                    experiments_loaded = True
                else:
                    print(
                        "   ⚠️  No valid runs with group_name found, using table fallback"
                    )
                    experiments_loaded = False

        except Exception as e:
            print(f"   ⚠️  Error using ExperimentTracking API: {str(e)[:200]}")
            print("   Will use table fallback")
            experiments_loaded = False

except Exception as e:
    print(f"   ⚠️  ML Experiments not available: {str(e)[:200]}")
    print("   Will use table fallback")
    experiments_loaded = False

# %% [markdown]
# ### 2b. Fallback to table

# %%
if not experiments_loaded or len(hyperparams_by_group) < len(expected_groups):
    print("\n📋 Fallback: HYPERPARAMETER_RESULTS")

    table_exists = False
    try:
        check_table = session.sql(
            """
            SELECT COUNT(*) as CNT 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = 'SC_MODELS_BMX' 
            AND TABLE_NAME = 'HYPERPARAMETER_RESULTS'
            AND TABLE_CATALOG = 'BD_AA_DEV'
            """
        ).collect()
        table_exists = check_table[0]["CNT"] > 0
    except:
        table_exists = False

    if table_exists:
        hyperparams_df = session.sql(
            f"""
            WITH latest_searches AS (
                SELECT 
                    group_name,
                    search_id,
                    algorithm,
                    best_params,
                    best_cv_rmse,
                    val_rmse,
                    val_mae,
                    created_at,
                    ROW_NUMBER() OVER (PARTITION BY group_name ORDER BY created_at DESC) AS rn
                FROM {HYPERPARAMETER_RESULTS_TABLE}
                WHERE group_name IS NOT NULL
            )
            SELECT 
                group_name,
                search_id,
                best_params,
                best_cv_rmse,
                val_rmse,
                val_mae
            FROM latest_searches
            WHERE rn = 1
            ORDER BY group_name
        """
        )

        hyperparams_results = hyperparams_df.collect()

        if len(hyperparams_results) > 0:
            print(f"   ✅ Loaded {len(hyperparams_results)} groups from table")

            for result in hyperparams_results:
                group_name = result["GROUP_NAME"]
                best_params_json = result["BEST_PARAMS"]

                if isinstance(best_params_json, str):
                    best_params = json.loads(best_params_json)
                else:
                    best_params = best_params_json

                if group_name not in hyperparams_by_group:
                    alg = result.get("ALGORITHM") or GROUP_MODEL.get(
                        group_name, _DEFAULT_MODEL
                    )
                    hyperparams_by_group[group_name] = {
                        "params": best_params,
                        "val_rmse": result["VAL_RMSE"],
                        "search_id": result["SEARCH_ID"],
                        "algorithm": alg,
                    }

                    print(f"\n   {group_name}:")
                    print(f"      Algorithm: {alg}")
                    print(f"      Val RMSE: {result['VAL_RMSE']:.4f}")
                    print(f"      Search ID: {result['SEARCH_ID']}")
                    print(f"      Source: Table (fallback)")
        else:
            print("   ⚠️  Table exists but has no results")
    else:
        print("   ⚠️  Table does not exist (this is OK if using ML Experiments)")

# %% [markdown]
# ### 2c. Defaults and validation

# %%
if len(hyperparams_by_group) == 0:
    raise ValueError(
        "No hyperparameter results found in Experiments or table! Please run 03_hyperparameter_search.py first"
    )

print(
    f"\n✅ Total loaded hyperparameters: {len(hyperparams_by_group)}/{len(expected_groups)} groups"
)

DEFAULT_PARAMS_BY_MODEL = {
    "XGBRegressor": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
    },
    "LGBMRegressor": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "min_child_samples": 20,
    },
    "SGDRegressor": {
        "alpha": 0.0001,
        "max_iter": 2000,
        "tol": 1e-3,
        "eta0": 0.01,
    },
}

print(f"\n📋 Defaults per model: {list(DEFAULT_PARAMS_BY_MODEL.keys())}")
print(f"🔍 Validating coverage...")
groups_with_hyperparams = set(hyperparams_by_group.keys())
groups_without_hyperparams = set(expected_groups) - groups_with_hyperparams

if groups_without_hyperparams:
    print(
        f"⚠️  WARNING: {len(groups_without_hyperparams)} groups will use default hyperparameters:"
    )
    for group in sorted(groups_without_hyperparams):
        print(f"      - {group}")
else:
    print(f"✅ All {len(expected_groups)} groups have optimized hyperparameters!")

# %% [markdown]
# ## 3. Training Data

# %%
print("\n🏪 Loading training data...")

if USE_CLEANED_TABLES:
    print("📊 Loading from cleaned table: TRAIN_DATASET_CLEANED")
    training_df = session.table(TRAIN_TABLE_CLEANED)
    print(f"\n✅ Training data loaded from cleaned table")
    print(f"   Total records: {training_df.count():,}")
    print(f"   Columns: {len(training_df.columns)}")
else:
    # Prefer materialized features table (without Dynamic Tables).
    # If it fails due to permissions/non-existence, fallback to cleaned table.
    try:
        # Initialize Feature Store (even though we don't use FeatureView)
        _fs = FeatureStore(
            session=session,
            database=DATABASE,
            name=FEATURES_SCHEMA,
            default_warehouse=DEFAULT_WAREHOUSE,
        )
        print("✅ Feature Store initialized (without FeatureView)")

        print(f"📊 Loading features from table: {FEATURES_TABLE}")
        features_df = session.table(FEATURES_TABLE)

        print(f"⏳ Loading target variable and {STATS_NTILE_GROUP_COL} from training table...")
        target_df = session.table(TRAIN_TABLE_CLEANED).select(
            "CUSTOMER_ID", "BRAND_PRES_RET", "PROD_KEY", "WEEK", TARGET_COLUMN, STATS_NTILE_GROUP_COL
        )

        print("⏳ Joining features with target...")
        training_df = features_df.join(
            target_df, on=["CUSTOMER_ID", "BRAND_PRES_RET", "PROD_KEY", "WEEK"], how="inner"
        )

        print(f"\n✅ Training data loaded from features table + target")
        print(f"   Total records: {training_df.count():,}")
        print(f"   Columns: {len(training_df.columns)}")
    except Exception as e:
        print(
            f"⚠️  Could not load/join features table ({FEATURES_TABLE}): {str(e)[:200]}"
        )
        print("   Falling back to TRAIN_DATASET_CLEANED")
        training_df = session.table(TRAIN_TABLE_CLEANED)
        print(f"\n✅ Training data loaded from cleaned table (fallback)")
        print(f"   Total records: {training_df.count():,}")
        print(f"   Columns: {len(training_df.columns)}")

PARTITION_COL = next(
    (c for c in training_df.columns if c.upper() == STATS_NTILE_GROUP_COL),
    STATS_NTILE_GROUP_COL,
)
print(f"\n📌 Partition column: '{PARTITION_COL}'")
print("\n📊 Rows per group:")
group_counts = (
    training_df.group_by(PARTITION_COL).count().sort(PARTITION_COL)
)
group_counts.show(n=20)

if MMT_SAMPLE_FRACTION is not None and 0 < MMT_SAMPLE_FRACTION < 1:
    n_before = training_df.count()
    training_df = training_df.sample(frac=MMT_SAMPLE_FRACTION)
    n_after = training_df.count()
    print(f"\n⚠️  MMT in TEST mode: using {MMT_SAMPLE_FRACTION*100:.0f}% of data ({n_after:,} of {n_before:,} rows)")

# %% [markdown]
# ## 4. MMT Training Function

# %%
def _get_target_column(df):
    for c in df.columns:
        if str(c).upper() == TARGET_COLUMN:
            return c
    return TARGET_COLUMN.lower()


def _get_feature_cols_numeric(df, excluded_cols, target_col):
    """Numeric columns only (same as script 03): Snowflake ML requires int/float/bool."""
    # excluded_cols is already in UPPER CASE, target_col may vary
    excluded_upper = {col.upper() if isinstance(col, str) else str(col).upper() for col in excluded_cols}
    excluded_upper.add(str(target_col).upper())
    return [
        col
        for col in df.columns
        if str(col).upper() not in excluded_upper
        and getattr(df[col].dtype, "kind", "O") in "iufb"
    ]


def train_segment_model(data_connector, context):
    import pandas as pd
    from snowflake.ml.modeling.xgboost import XGBRegressor
    from snowflake.ml.modeling.lightgbm import LGBMRegressor
    from snowflake.ml.modeling.linear_model import SGDRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np

    segment_name = context.partition_id
    print(f"\n{'='*80}")
    print(f"🚀 Training model for {segment_name}")
    print(f"{'='*80}")

    # NOTE: DataConnector in MMT only provides to_pandas() method, not direct Snowpark DataFrame access
    # This means we need to convert to pandas, but we can optimize by doing the split efficiently
    # For very large partitions, consider using MMT_SAMPLE_FRACTION to reduce partition size
    df = data_connector.to_pandas()
    print(f"📊 Data shape: {df.shape}")
    
    target_col = _get_target_column(df)
    feature_cols = _get_feature_cols_numeric(df, EXCLUDED_COLS, target_col)
    if len(feature_cols) < 5:
        excluded_upper = {col for col in EXCLUDED_COLS}
        excluded_upper.add(str(target_col).upper())
        feature_cols = [c for c in df.columns if str(c).upper() not in excluded_upper]
    
    # Get statistics
    X = df[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)

    print(f"   Features: {len(feature_cols)}")
    print(f"   Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"   Target mean: {y.mean():.2f}")

    # Split using pandas (original behavior)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training set: {X_train.shape[0]:,} samples")
    print(f"   Test set: {X_test.shape[0]:,} samples")

    # Prepare data
    train_dataset = X_train.copy()
    for c in feature_cols:
        train_dataset[c] = np.asarray(train_dataset[c], dtype=np.float64)
    train_dataset[target_col] = np.asarray(y_train, dtype=np.float64)
    test_features = X_test.copy()
    for c in feature_cols:
        test_features[c] = np.asarray(test_features[c], dtype=np.float64)

    model_type = GROUP_MODEL.get(segment_name, _DEFAULT_MODEL)
    if segment_name in hyperparams_by_group:
        algorithm = hyperparams_by_group[segment_name].get("algorithm")
        if algorithm:
            model_type = algorithm
        group_params = hyperparams_by_group[segment_name]["params"]
        search_id = hyperparams_by_group[segment_name]["search_id"]
        val_rmse = hyperparams_by_group[segment_name]["val_rmse"]
        print(f"\n   ✅ Using OPTIMIZED hyperparameters from script 03")
        print(f"      Model: {model_type}")
        print(f"      Search ID: {search_id}")
        print(f"      Validation RMSE (from search): {val_rmse:.4f}")
    else:
        group_params = DEFAULT_PARAMS_BY_MODEL.get(
            model_type, DEFAULT_PARAMS_BY_MODEL["XGBRegressor"]
        )
        print(
            f"\n   ⚠️  Using DEFAULT hyperparameters for {model_type} (no search results for {segment_name})"
        )

    def _to_native(v):
        """Same as script 03: numpy -> Python native; numeric string -> float/int."""
        if hasattr(v, "item"):
            return v.item()
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, (np.integer, np.floating)):
            return int(v) if isinstance(v, np.integer) else float(v)
        if isinstance(v, str):
            v = v.strip()
            try:
                f = float(v)
                return int(f) if f == int(f) else f
            except (ValueError, TypeError):
                return v
        return v

    int_params = ("n_estimators", "max_depth", "num_leaves", "min_child_weight", "min_child_samples", "max_iter")
    float_params = ("alpha", "learning_rate", "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda", "tol", "eta0")
    defaults = DEFAULT_PARAMS_BY_MODEL.get(model_type, DEFAULT_PARAMS_BY_MODEL["XGBRegressor"])
    model_params = {}
    for k, v in group_params.items():
        vn = _to_native(v)
        try:
            if k in int_params:
                model_params[k] = int(vn) if isinstance(vn, (int, float, np.integer, np.floating)) else defaults.get(k, vn)
            elif k in float_params:
                model_params[k] = float(vn) if isinstance(vn, (int, float, np.integer, np.floating)) else defaults.get(k, vn)
            else:
                model_params[k] = vn
        except (TypeError, ValueError):
            model_params[k] = defaults.get(k, vn)
    model_params["random_state"] = 42

    MODEL_CLASSES = {
        "XGBRegressor": XGBRegressor,
        "LGBMRegressor": LGBMRegressor,
        "SGDRegressor": SGDRegressor,
    }
    ModelClass = MODEL_CLASSES.get(model_type, XGBRegressor)
    if model_type == "XGBRegressor":
        model_params["n_jobs"] = -1
        model_params["objective"] = "reg:squarederror"
        model_params["eval_metric"] = "rmse"
    elif model_type == "LGBMRegressor":
        model_params["n_jobs"] = -1
        model_params["verbosity"] = -1
    elif model_type == "SGDRegressor":
        model_params.setdefault("penalty", "l2")
        model_params.setdefault("learning_rate", "invscaling")

    print(f"\n   Training {model_type} with {len(model_params)} hyperparameters...")
    model = ModelClass(
        input_cols=feature_cols, label_cols=[target_col], **model_params
    )
    model.fit(train_dataset)

    pred_result = model.predict(test_features)
    pred_df = pred_result.to_pandas() if hasattr(pred_result, "to_pandas") else pred_result
    out_col = model.get_output_cols()[0]
    y_pred = np.asarray(pred_df[out_col])

    # Regression metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # WAPE: sum(|y - y_hat|) / sum(|y|)
    abs_errors = np.abs(y_test - y_pred)
    denom_wape = np.sum(np.abs(y_test))
    wape = float(abs_errors.sum() / denom_wape) if denom_wape > 0 else 0.0

    # MAPE: mean(|y - y_hat| / |y|) * 100, ignoring targets 0
    non_zero_mask = np.abs(y_test) > 1e-8
    if non_zero_mask.any():
        mape = float(
            (np.abs(y_test[non_zero_mask] - y_pred[non_zero_mask]) / np.abs(y_test[non_zero_mask])).mean()
            * 100.0
        )
    else:
        mape = 0.0

    print(f"\n   ✅ Model trained")
    print(f"      RMSE: {rmse:.2f}")
    print(f"      MAE: {mae:.2f}")
    print(f"      WAPE: {wape:.4f}")
    print(f"      MAPE: {mape:.2f}%")
    print(f"{'='*80}\n")

    model.rmse = rmse
    model.mae = mae
    model.wape = wape
    model.mape = mape
    model.training_samples = X_train.shape[0]
    model.test_samples = X_test.shape[0]
    model.feature_cols = feature_cols
    model.hyperparameters = model_params
    model.segment = segment_name
    model.group_name = segment_name

    return model


# %% [markdown]
# ## 5. Escalar cluster, Ray Dashboard, MMT

# %%
try:
    from snowflake.ml.runtime_cluster import scale_cluster
    scale_cluster(expected_cluster_size=CLUSTER_SIZE_MMT, options={"block_until_min_cluster_size": CLUSTER_SIZE_MIN_MMT})
    print(f"✅ Cluster scaled to {CLUSTER_SIZE_MMT} nodes")
except Exception as e:
    print(f"⚠️ scale_cluster: {str(e)[:150]}")

try:
    from snowflake.ml.runtime_cluster import get_ray_dashboard_url
    print(f"✅ Ray Dashboard: {get_ray_dashboard_url()}")
except Exception as e:
    print(f"⚠️ Ray Dashboard: {str(e)[:100]}")

# %%
start_time = time.time()
trainer = ManyModelTraining(train_segment_model, MMT_STAGE)
training_run = trainer.run(
    partition_by=PARTITION_COL,
    snowpark_dataframe=training_df,
    run_id=f"uni_box_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)

print(f"\n✅ Run ID: {training_run.run_id}\n")

# %% [markdown]
# ### 5d. Wait for MMT (optional; if partition_details fails, exit loop)

# %%
import time as time_module
MMT_MAX_WAIT = 600
MMT_CHECK_INTERVAL = 30

elapsed = 0
completed = False
run_start = start_time

while elapsed < MMT_MAX_WAIT:
    time_module.sleep(MMT_CHECK_INTERVAL)
    elapsed += MMT_CHECK_INTERVAL

    try:
        details = training_run.partition_details
    except Exception as e:
        print(f"\n⚠️  partition_details failed: {str(e)[:180]}")
        print("   Stop waiting. Check Ray Dashboard or run 6/7 later.")
        break

    total_count = len(details)
    done_count = sum(1 for pid in details if details[pid].status.name == "DONE")
    failed_count = sum(1 for pid in details if details[pid].status.name == "FAILED")
    pending_count = total_count - done_count - failed_count
    print(
        f"⏱️  {elapsed}s - OK: {done_count} | FAILED: {failed_count} | pending: {pending_count}",
        end="\r",
    )

    if done_count + failed_count == total_count:
        print("\n✅ All models completed!" + " " * 30)
        completed = True
        break

if not completed:
    print("\n⏱️  Timeout. Training may continue in background; check Ray Dashboard or run 6/7 later.")
    try:
        stage_files = session.sql(
            f"LIST @{MMT_STAGE} PATTERN='.*{training_run.run_id}.*'"
        ).collect()
        if len(stage_files) >= 16:
            print(f"\n✅ {len(stage_files)} files in stage - training likely completed.")
            completed = True
    except Exception:
        pass
else:
    print("\n✅ TRAINING COMPLETE")
end_time = time.time()
print(f"\n⏱️  {((end_time - run_start) / 60):.2f} min")

try:
    from snowflake.ml.runtime_cluster import scale_cluster
    scale_cluster(expected_cluster_size=CLUSTER_SIZE_DOWN)
    print(f"✅ Cluster scaled down to {CLUSTER_SIZE_DOWN} node")
except Exception as e:
    print(f"⚠️ scale down: {str(e)[:120]}")

# %% [markdown]
# ## 6. Results per Partition

# %%
try:
    partition_details = training_run.partition_details
except Exception as e:
    partition_details = {}
    print(f"⚠️ partition_details failed: {str(e)[:200]}. Re-run from §3, then 5c→5d→6.")

done_ids = []
failed_ids = []
pending_ids = []
for partition_id in partition_details:
    details = partition_details[partition_id]
    st = details.status.name
    if st == "DONE":
        done_ids.append(partition_id)
        try:
            model = training_run.get_model(partition_id)
            print(f"\n✅ {partition_id}: RMSE={model.rmse:.2f}, MAE={model.mae:.2f}, samples={model.training_samples:,}")
        except Exception as e:
            print(f"\n⚠️  {partition_id}: DONE pero no se pudo cargar - {str(e)[:100]}")
    elif st == "FAILED":
        failed_ids.append(partition_id)
        print(f"\n❌ {partition_id}: FAILED")
        try:
            logs = getattr(details, "logs", None)
            if logs and "Error:" in logs:
                err_line = next((l for l in logs.split("\n") if "Error:" in l), None)
                if err_line:
                    print(f"   {err_line.strip()[:200]}")
        except Exception:
            pass
    else:
        pending_ids.append(partition_id)
        print(f"\n⏳ {partition_id}: {st}")
print(f"\n--- Resumen: {len(done_ids)} OK, {len(failed_ids)} FAILED, {len(pending_ids)} pendientes ---")

# %% [markdown]
# ## 7. Register Models in Registry

# %%
version_date = datetime.now().strftime("%Y%m%d_%H%M")
registered_models = {}
try:
    _reg_partitions = training_run.partition_details
except Exception as e:
    _reg_partitions = {}
    print(f"⚠️ partition_details: {str(e)[:180]}")

for partition_id in _reg_partitions:
    details = _reg_partitions[partition_id]

    if details.status.name == "DONE":
        try:
            model = training_run.get_model(partition_id)

            model_name = f"uni_box_regression_{partition_id.lower()}"
            group_search_id = None
            group_hyperparams = None
            group_algorithm = GROUP_MODEL.get(partition_id, _DEFAULT_MODEL)
            if partition_id in hyperparams_by_group:
                group_search_id = hyperparams_by_group[partition_id]["search_id"]
                group_hyperparams = hyperparams_by_group[partition_id]["params"]
                alg = hyperparams_by_group[partition_id].get("algorithm")
                if alg:
                    group_algorithm = alg

            print(f"\nRegistrando {partition_id}...")
            model_metrics = {
                "rmse": float(model.rmse),
                "mae": float(model.mae),
                "wape": float(model.wape),
                "mape": float(model.mape),
                "training_samples": int(model.training_samples),
                "test_samples": int(model.test_samples),
                "algorithm": group_algorithm,
                "group": partition_id,
                "hyperparameter_search_id": group_search_id or "default",
            }

            if feature_version_id:
                model_metrics["feature_version_id"] = feature_version_id
            if feature_snapshot_at is not None:
                model_metrics["feature_snapshot_at"] = str(feature_snapshot_at)
            model_metrics["feature_table_name"] = FEATURES_TABLE

            if group_hyperparams:
                for key, value in group_hyperparams.items():
                    if isinstance(value, (int, float)):
                        model_metrics[f"hyperparameter_{key}"] = (
                            float(value) if isinstance(value, float) else int(value)
                        )
                model_metrics["hyperparameters"] = json.dumps(
                    {
                        k: float(v) if isinstance(v, (int, float)) else v
                        for k, v in group_hyperparams.items()
                    }
                )

            mv = registry.log_model(
                model,
                model_name=model_name,
                version_name=f"v_{version_date}",
                comment=f"{group_algorithm} regression model for uni_box_week - Group: {partition_id}",
                metrics=model_metrics,
                task=task.Task.TABULAR_REGRESSION,
            )

            registered_models[partition_id] = {
                "model_name": model_name,
                "version": f"v_{version_date}",
                "model_version": mv,
            }

            print(f"✅ {partition_id}: {model_name} v_{version_date}")
            print(
                f"   RMSE: {model.rmse:.2f}, MAE: {model.mae:.2f}, "
                f"WAPE: {model.wape:.4f}, MAPE: {model.mape:.2f}%"
            )
            
            # Set PRODUCTION alias to this version (move from previous version if needed)
            try:
                model_fqn = f"{DATABASE}.{MODELS_SCHEMA}.{model_name}"
                new_version_name = f"v_{version_date}"

                # Try to remove PRODUCTION alias from previous version (if any)
                try:
                    session.sql(f"ALTER MODEL {model_fqn} VERSION PRODUCTION UNSET ALIAS").collect()
                except Exception:
                    # It is fine if there was no previous PRODUCTION alias
                    pass

                # Assign PRODUCTION alias to the newly logged version
                session.sql(
                    f"ALTER MODEL {model_fqn} VERSION {new_version_name} SET ALIAS=PRODUCTION"
                ).collect()
                print("   ✅ PRODUCTION alias set")
            except Exception as e:
                print(f"   ⚠️  Error setting PRODUCTION alias: {str(e)[:100]}")

        except Exception as e:
            print(f"❌ Error registering model: {str(e)[:200]}")

print(f"\n✅ {len(registered_models)} model(s) registered successfully!")

# %% [markdown]
# ## 8. Verify PRODUCTION Alias

# %%
# PRODUCTION alias is already set during registration (section 7)
# This section verifies all models have the alias
print("\n📋 Verifying PRODUCTION alias for all registered models...")
for partition_id, model_info in registered_models.items():
    model_name = model_info["model_name"]
    version = model_info["version"]
    try:
        model_ref = registry.get_model(model_name)
        prod_version = model_ref.version("PRODUCTION")
        # ModelVersion may expose the version name under different attributes depending on library version
        version_name = getattr(prod_version, "name", getattr(prod_version, "version_name", str(prod_version)))
        print(f"✅ {model_name}: PRODUCTION → {version_name}")
    except Exception as e:
        print(f"⚠️  {model_name}: PRODUCTION alias not found - {str(e)[:100]}")

# %% [markdown]
# ## 9. Resumen

# %%
_elapsed = (time.time() - start_time) / 60
print(f"\n✅ MMT: {len(registered_models)}/16 models | {_elapsed:.2f} min")
if registered_models:
    for pid in sorted(registered_models.keys()):
        try:
            m = training_run.get_model(pid)
            print(
                f"   {pid}: RMSE={m.rmse:.2f}, MAE={m.mae:.2f}, "
                f"WAPE={m.wape:.4f}, MAPE={m.mape:.2f}%"
            )
        except Exception:
            pass
print("   Next: 05_create_partitioned_model.py → 06_partitioned_inference_batch.py")

