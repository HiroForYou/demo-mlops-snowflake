# %% [markdown]
# # Migration: Hyperparameter Search (LGBM/XGB + Snowflake ML Tune) - Per Group - Bayesian Search
#
# ## Overview
# This script performs hyperparameter optimization using Snowflake ML's tune.search for regression (LGBM, XGBoost per group).
# **Runs HPO per group in a sequential loop (no MMT) to avoid Ray serialization issues; each group runs its own Bayesian Search.**
#
# ## What We'll Do:
# 1. Load cleaned training data with stats_ntile_group
# 2. Get all 16 unique groups
# 3. For each group (loop): load group data, run Bayesian Search with snowflake.ml.modeling.tune, save best hyperparameters to SC_MODELS_BMX.HYPERPARAMETER_RESULTS
# 4. Generate summary of all hyperparameter results

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.ml.feature_store import FeatureStore
from snowflake.ml.modeling.tune import (
    Tuner,
    TunerConfig,
    get_tuner_context,
    uniform,
)
from snowflake.ml.modeling.tune.search import BayesOpt
from snowflake.ml.data.data_connector import DataConnector
from snowflake.ml.experiment import ExperimentTracking
import numpy as np
from datetime import datetime
import json
import time

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

# Hyperparameter search configuration
NUM_TRIALS = 15
MAX_CONCURRENT_TRIALS = 4
SAMPLE_RATE_PER_GROUP = 0.2

# Cluster scaling configuration
CLUSTER_SIZE_HPO = 5
CLUSTER_SIZE_MIN_HPO = 2
CLUSTER_SIZE_DOWN = 1

# Set context
session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()

print(f"✅ Connected to Snowflake")
print(f"   Database: {session.get_current_database()}")
print(f"   Schema: {session.get_current_schema()}")

# Resolve active feature version (for traceability)
feature_version_id = None
feature_snapshot_at = None
try:
    version_rows = session.sql(
        f"""
        SELECT FEATURE_VERSION_ID, FEATURE_SNAPSHOT_AT, CREATED_AT
        FROM {FEATURE_VERSIONS_TABLE}
        WHERE FEATURE_TABLE_NAME = 'UNI_BOX_FEATURES'
          AND IS_ACTIVE = TRUE
        ORDER BY CREATED_AT DESC
        LIMIT 1
    """
    ).collect()
    if version_rows:
        feature_version_id = version_rows[0]["FEATURE_VERSION_ID"]
        feature_snapshot_at = version_rows[0]["FEATURE_SNAPSHOT_AT"]
        print("\n📌 Active feature version for training:")
        print(f"   FEATURE_VERSION_ID: {feature_version_id}")
        print(f"   FEATURE_SNAPSHOT_AT: {feature_snapshot_at}")
    else:
        print("\n⚠️  No active feature version found in FEATURE_VERSIONS; proceeding without version metadata")
except Exception as e:
    print(f"\n⚠️  Could not resolve feature version: {str(e)[:200]}")

# Configuration:
# - If you don't have permissions for FeatureView/Dynamic Tables, use cleaned tables.
# - Keep the flag but add automatic fallback if Feature Store mode fails.
USE_CLEANED_TABLES = (
    False  # True = TRAIN_DATASET_CLEANED, False = try Feature Store
)

# Single object: group -> Snowflake ML class name (snowflake.ml.modeling.*).
# All groups use boosting models (XGB/LGBM) for regression.
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
# ## 1. Get All Groups and Load Training Data

# %%
print("\n" + "=" * 80)
print("📊 GETTING ALL GROUPS")
print("=" * 80)

# Get all unique groups from cleaned training table
groups_df = session.sql(
    f"""
    SELECT DISTINCT {STATS_NTILE_GROUP_COL} AS GROUP_NAME
    FROM {TRAIN_TABLE_CLEANED}
    WHERE {STATS_NTILE_GROUP_COL} IS NOT NULL
    ORDER BY {STATS_NTILE_GROUP_COL}
"""
)

groups_list = [row["GROUP_NAME"] for row in groups_df.collect()]
print(f"\n✅ Found {len(groups_list)} groups:")
for i, group in enumerate(groups_list, 1):
    print(f"   {i:2d}. {group}")

if len(groups_list) != 16:
    print(f"\n⚠️  WARNING: Expected 16 groups, found {len(groups_list)}")
    print("   Continuing with available groups...")

# %% [markdown]
# ## 2. Load Training Data (Feature Store or Cleaned Tables)

# %%
if USE_CLEANED_TABLES:
    print("\n" + "=" * 80)
    print("📊 LOADING DATA FROM CLEANED TABLES (TESTING MODE)")
    print("=" * 80)

    # Load directly from cleaned training table (for testing purposes)
    print("⏳ Loading data from TRAIN_DATASET_CLEANED...")
    train_df = session.table(TRAIN_TABLE_CLEANED)

    total_rows = train_df.count()
    print(f"\n✅ Training data loaded from cleaned table")
    print(f"   Total rows: {total_rows:,}")
    print(f"   ⚠️  TESTING MODE: Using cleaned tables directly (not Feature Store)")
else:
    print("\n" + "=" * 80)
    print("🏪 FEATURE STORE MODE (WITHOUT FEATUREVIEW)")
    print("=" * 80)

    # Instead of FeatureView, we support a materialized features table created by `02_feature_store_setup.py`
    # (without Dynamic Tables). If it doesn't exist or fails, we fallback to cleaned tables.
    try:
        # Initialize Feature Store (even though we don't use FeatureView)
        _fs = FeatureStore(
            session=session,
            database=DATABASE,
            name=FEATURES_SCHEMA,
            default_warehouse=DEFAULT_WAREHOUSE,
        )
        print("✅ Feature Store initialized (without FeatureView)")

        print(f"⏳ Loading features from table: {FEATURES_TABLE} ...")
        features_df = session.table(FEATURES_TABLE)

        print(f"⏳ Loading target variable and {STATS_NTILE_GROUP_COL} from training table...")
        target_df = session.table(TRAIN_TABLE_CLEANED).select(
            "CUSTOMER_ID", "BRAND_PRES_RET", "PROD_KEY", "WEEK", TARGET_COLUMN, STATS_NTILE_GROUP_COL
        )

        print("⏳ Joining features with target...")
        train_df = features_df.join(
            target_df, on=["CUSTOMER_ID", "BRAND_PRES_RET", "PROD_KEY", "WEEK"], how="inner"
        )

        total_rows = train_df.count()
        print(f"\n✅ Training data loaded from features table + target")
        print(f"   Total rows: {total_rows:,}")
    except Exception as e:
        print(
            f"⚠️  Could not load/join features table ({FEATURES_TABLE}): {str(e)[:200]}"
        )
        print(
            "   Falling back to TRAIN_DATASET_CLEANED (USE_CLEANED_TABLES=True behavior)"
        )
        train_df = session.table(TRAIN_TABLE_CLEANED)
        total_rows = train_df.count()
        print(f"\n✅ Training data loaded from cleaned table (fallback)")
        print(f"   Total rows: {total_rows:,}")

# %% [markdown]
# ## 3. Define Hyperparameter Search Space

# %%
print("\n" + "=" * 80)
print("🎯 DEFINING HYPERPARAMETER SEARCH SPACE")
print("=" * 80)

# Search spaces per model type (LGBM, XGBoost).
# BayesOpt requires continuous search spaces and only uniform() sampling (see Snowflake docs:
# https://docs.snowflake.com/en/developer-guide/snowflake-ml/container-hpo).
# Integer params use uniform(low, high) and are cast to int inside the training function.
SEARCH_SPACES = {
    "XGBRegressor": {
        "n_estimators": uniform(50, 300),
        "max_depth": uniform(3, 10),
        "learning_rate": uniform(0.01, 0.3),
        "subsample": uniform(0.6, 1.0),
        "colsample_bytree": uniform(0.6, 1.0),
        "min_child_weight": uniform(1, 7),
        "gamma": uniform(0, 0.5),
        "reg_alpha": uniform(0, 1),
        "reg_lambda": uniform(0, 1),
    },
    "LGBMRegressor": {
        "n_estimators": uniform(50, 300),
        "max_depth": uniform(3, 10),
        "learning_rate": uniform(0.01, 0.3),
        "num_leaves": uniform(20, 150),
        "subsample": uniform(0.6, 1.0),
        "colsample_bytree": uniform(0.6, 1.0),
        "reg_alpha": uniform(0, 1),
        "reg_lambda": uniform(0, 1),
        "min_child_samples": uniform(5, 50),
    },
}

# Parameters that must be integers (sampled with uniform() for BayesOpt, cast in train_func)
INT_PARAMS = {"n_estimators", "max_depth", "min_child_weight", "num_leaves", "min_child_samples"}

print("\n📋 Hyperparameter Search Spaces (per model type):")
for model_type, search_space in SEARCH_SPACES.items():
    print(f"   {model_type}: {list(search_space.keys())}")
print("\n📋 XGBRegressor search space (BayesOpt: uniform only, int params cast in train_func):")
for param, dist in SEARCH_SPACES["XGBRegressor"].items():
    if hasattr(dist, "low") and hasattr(dist, "high"):
        print(f"   {param}: uniform({dist.low}, {dist.high})")

print(f"\n🔢 Bayesian Search trials per group: {NUM_TRIALS}")
print(f"   Max concurrent trials per group: {MAX_CONCURRENT_TRIALS}")
print(f"📊 Sample rate per group: {SAMPLE_RATE_PER_GROUP*100:.0f}%")
if SAMPLE_RATE_PER_GROUP < 1.0:
    print(
        f"   ⚠️  Using {SAMPLE_RATE_PER_GROUP*100:.0f}% of data - consider using 1.0 (full group) for better results"
    )
else:
    print(f"   ✅ Using full group data for optimal hyperparameter search")

# %% [markdown]
# ### 3c. (Optional) View Ray Dashboard
#
# Use this cell in Snowflake Notebooks to get the Ray Dashboard URL
# for the current runtime. Copy and paste the URL in your browser.

# %%
try:
    from snowflake.ml.runtime_cluster import get_ray_dashboard_url

    dashboard_url = get_ray_dashboard_url()
    print(f"✅ Access the Ray Dashboard here: {dashboard_url}")
except Exception as e:
    print("⚠️ Could not get Ray Dashboard URL.")
    print(f"   Detail: {str(e)[:200]}")

# %% [markdown]
# ## 4. Scale Cluster for HPO

# %%
print("\n" + "=" * 80)
print("📈 SCALING CLUSTER FOR HPO")
print("=" * 80)

try:
    from snowflake.ml.runtime_cluster import scale_cluster

    print(f"⏳ Scaling cluster to {CLUSTER_SIZE_HPO} containers...")
    scale_cluster(
        expected_cluster_size=CLUSTER_SIZE_HPO,
        options={
            "block_until_min_cluster_size": CLUSTER_SIZE_MIN_HPO
        }
    )
    print(f"✅ Cluster scaled to {CLUSTER_SIZE_HPO} containers")
except Exception as e:
    print(f"⚠️  Error scaling cluster: {str(e)[:200]}")
    print("   Continuing with current cluster...")

# %% [markdown]
# ## 5. Perform Hyperparameter Search Per Group

# %% [markdown]
# ### 5a. Setup: ML Experiments, table (if applicable), features and all_results

# %%
print("\n" + "=" * 80)
print("🔍 PERFORMING HYPERPARAMETER SEARCH PER GROUP")
print("=" * 80)

# Initialize ML Experiments for hyperparameter tracking FIRST
print("\n" + "=" * 80)
print("🔬 INITIALIZING ML EXPERIMENTS")
print("=" * 80)

# Initialize experiment_name in global scope (multi-model: LGBMRegressor, XGBRegressor, SGDRegressor)
experiment_name = (
    f"hyperparameter_search_regression_{datetime.now().strftime('%Y%m%d')}"
)

try:
    exp_tracking = ExperimentTracking(session)
    exp_tracking.set_experiment(experiment_name)
    print(f"✅ Experiment created: {experiment_name}")
    experiments_available = True
except Exception as e:
    print(f"⚠️  ML Experiments not available: {str(e)[:200]}")
    print("   Will continue with table-based storage only")
    exp_tracking = None
    experiments_available = False

# Create results table ONLY if ML Experiments is not available
# If Experiments is available, we don't need this table
if not experiments_available:
    print("\n📋 Creating HYPERPARAMETER_RESULTS table (Experiments not available)")
    session.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {HYPERPARAMETER_RESULTS_TABLE} (
            search_id VARCHAR,
            group_name VARCHAR,
            algorithm VARCHAR,
            best_params VARIANT,
            best_cv_rmse FLOAT,
            best_cv_mae FLOAT,
            val_rmse FLOAT,
            val_mae FLOAT,
            n_iter INTEGER,
            sample_size INTEGER,
            feature_version_id VARCHAR,
            feature_snapshot_at TIMESTAMP_NTZ,
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """
    ).collect()
    # Ensure new columns exist if table was created without them in the past (one ALTER per column)
    for col_def in ["feature_version_id VARCHAR", "feature_snapshot_at TIMESTAMP_NTZ"]:
        try:
            session.sql(
                f"ALTER TABLE {HYPERPARAMETER_RESULTS_TABLE} ADD COLUMN IF NOT EXISTS {col_def}"
            ).collect()
        except Exception:
            pass
    print("   ✅ Table created/updated (will be used as primary storage)")
else:
    print("\n📋 Skipping table creation (using ML Experiments as primary storage)")


# %% [markdown]
# ### 4b. Helper Functions: Target and Numeric Features

# %%
def _get_target_column(df):
    """Return the target column name in df (case-insensitive match)."""
    for c in df.columns:
        if str(c).upper() == TARGET_COLUMN:
            return c
    return TARGET_COLUMN.lower()


def _get_feature_cols_numeric(df, excluded_cols, target_col):
    """Numeric feature columns: excludes metadata/target (case-insensitive) and only int/float/bool (Snowflake ML)."""
    # excluded_cols is already in UPPER CASE, target_col may vary
    excluded_upper = {col.upper() if isinstance(col, str) else str(col).upper() for col in excluded_cols}
    excluded_upper.add(str(target_col).upper())
    return [
        col
        for col in df.columns
        if str(col).upper() not in excluded_upper
        and getattr(df[col].dtype, "kind", "O") in "iufb"
    ]


# Get feature columns from first group (all groups should have same features)
# Use limit(1).to_pandas() only for schema inspection - minimal memory impact
stats_ntile_col = next((c for c in train_df.columns if c.upper() == STATS_NTILE_GROUP_COL), STATS_NTILE_GROUP_COL)
sample_group_df = train_df.filter(train_df[stats_ntile_col] == groups_list[0])
# Only convert 1 row to pandas for schema inspection (minimal memory)
sample_pandas = sample_group_df.limit(1).to_pandas()
target_col_sample = _get_target_column(sample_pandas)
feature_cols = _get_feature_cols_numeric(sample_pandas, EXCLUDED_COLS, target_col_sample)

print(f"\n📋 Features ({len(feature_cols)}):")
for col in sorted(feature_cols):
    print(f"   - {col}")

# Dictionary to store all results (filled in loop 5c)
all_results = {}

# %% [markdown]
# ### 4c. Training Function for the Tuner

# %%
def create_train_func_for_tuner(feature_cols, model_type, target_col):
    """
    Create a training function for the Tuner (XGBRegressor or LGBMRegressor).
    Called for each trial with different hyperparameters; model_type selects the regressor class.

    Args:
        feature_cols: List of feature column names
        model_type: "XGBRegressor" or "LGBMRegressor"
        target_col: Actual target column name in dataset (e.g. uni_box_week or UNI_BOX_WEEK)

    Returns:
        train_func: Function that can be used by Tuner
    """

    def train_func():
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import numpy as np

        tuner_context = get_tuner_context()
        params = tuner_context.get_hyper_params()
        dm = tuner_context.get_dataset_map()

        train_pd = dm["train"].to_pandas()
        test_pd = dm["test"].to_pandas()
        # Snowflake ML uses fit(dataset) with a single DataFrame; input_cols and label_cols on the model
        train_dataset = train_pd[feature_cols + [target_col]].fillna(0)
        test_features = test_pd[feature_cols].fillna(0)
        y_val = test_pd[target_col].fillna(0).values

        model_params = params.copy()
        # BayesOpt uses only uniform(); cast integer hyperparameters (see Snowflake HPO docs)
        for k in INT_PARAMS:
            if k in model_params and model_params[k] is not None:
                model_params[k] = int(round(float(model_params[k])))
        model_params["random_state"] = 42

        if model_type == "XGBRegressor":
            from snowflake.ml.modeling.xgboost import XGBRegressor

            model_params["n_jobs"] = -1
            model_params["objective"] = "reg:squarederror"
            model_params["eval_metric"] = "rmse"
            model = XGBRegressor(
                input_cols=feature_cols, label_cols=[target_col], **model_params
            )
        elif model_type == "LGBMRegressor":
            from snowflake.ml.modeling.lightgbm import LGBMRegressor

            model_params["n_jobs"] = -1
            model_params["verbosity"] = -1
            model = LGBMRegressor(
                input_cols=feature_cols, label_cols=[target_col], **model_params
            )
        elif model_type == "SGDRegressor":
            from snowflake.ml.modeling.linear_model import SGDRegressor

            model_params.setdefault("penalty", "l2")
            model_params.setdefault("learning_rate", "invscaling")
            model = SGDRegressor(
                input_cols=feature_cols, label_cols=[target_col], **model_params
            )
        else:
            from snowflake.ml.modeling.xgboost import XGBRegressor

            model_params["n_jobs"] = -1
            model_params["objective"] = "reg:squarederror"
            model_params["eval_metric"] = "rmse"
            model = XGBRegressor(
                input_cols=feature_cols, label_cols=[target_col], **model_params
            )

        model.fit(train_dataset)
        pred_result = model.predict(test_features)
        pred_df = pred_result.to_pandas() if hasattr(pred_result, "to_pandas") else pred_result
        out_col = model.get_output_cols()[0]
        y_val_pred = np.asarray(pred_df[out_col])

        # Regression metrics
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)

        # WAPE: sum(|y - y_hat|) / sum(|y|)
        abs_errors = np.abs(y_val - y_val_pred)
        denom_wape = np.sum(np.abs(y_val))
        val_wape = float(abs_errors.sum() / denom_wape) if denom_wape > 0 else 0.0

        # MAPE: mean(|y - y_hat| / |y|) * 100, ignoring targets 0
        non_zero_mask = np.abs(y_val) > 1e-8
        if non_zero_mask.any():
            val_mape = float(
                (np.abs(y_val[non_zero_mask] - y_val_pred[non_zero_mask]) / np.abs(y_val[non_zero_mask])).mean()
                * 100.0
            )
        else:
            val_mape = 0.0

        tuner_context.report(
            metrics={
                "rmse": val_rmse,
                "mae": val_mae,
                "wape": val_wape,
                "mape": val_mape,
            },
            model=model,
        )

    return train_func


# %% [markdown]
# ### 4d. Hyperparameter Search for One Group

# %%
def run_hyperparameter_search_for_one_group(group_name, group_snowpark_df):
    """
    Run hyperparameter search for one group (HPO only, no MMT).
    Called from a loop in the main script to avoid Ray serialization issues.
    
    OPTIMIZED: Works with Snowpark DataFrame to avoid OOM, only converts to pandas
    when necessary for train_test_split and model training.

    Args:
        group_name: stats_ntile_group name (e.g. "GROUP_01").
        group_snowpark_df: Snowpark DataFrame with data for this group only.

    Returns:
        HyperparameterResult (best params, metrics) or DummyResult (skipped/failed).
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from snowflake.snowpark import functions as F
    import numpy as np

    print(f"\n{'='*80}")
    print(f"🔍 Hyperparameter Search for Group: {group_name}")
    print(f"{'='*80}")

    class DummyResult:
        def __init__(self):
            self.group_name = group_name
            self.skipped = True

    # Count records using Snowpark (avoids loading to memory)
    group_count = group_snowpark_df.count()
    print(f"📊 Group data: {group_count:,} records")

    if group_count < 50:
        print(
            f"⚠️  WARNING: Group has less than 50 records. Skipping hyperparameter search."
        )
        print(f"   Will use default hyperparameters for this group.")
        return DummyResult()

    # Sample data for this group using Snowpark (avoids loading full group to memory)
    if SAMPLE_RATE_PER_GROUP < 1.0:
        sampled_snowpark_df = group_snowpark_df.sample(frac=SAMPLE_RATE_PER_GROUP)
        sampled_count = sampled_snowpark_df.count()
        print(
            f"   Sampled: {sampled_count:,} records ({SAMPLE_RATE_PER_GROUP*100:.0f}% of {group_count:,} total)"
        )
    else:
        # Use full group for better hyperparameter search
        sampled_snowpark_df = group_snowpark_df
        sampled_count = group_count
        print(f"   Using full group: {sampled_count:,} records (100% of group)")

    # Get column names and target from Snowpark DataFrame (no pandas conversion yet)
    # Get a small sample to identify column names and types
    sample_row = sampled_snowpark_df.limit(1).to_pandas()
    target_col = _get_target_column(sample_row)
    feature_cols_list = _get_feature_cols_numeric(sample_row, EXCLUDED_COLS, target_col)

    # Check if we have enough data for train/val split
    if sampled_count < 20:
        print(f"⚠️  WARNING: Not enough data for train/val split. Skipping.")
        return DummyResult()

    # Prepare data in Snowpark: select features and target, fill nulls
    # This keeps data in Snowflake until we need it for train_test_split
    feature_cols_filled = [F.coalesce(F.col(c), F.lit(0)).alias(c) for c in feature_cols_list]
    target_col_filled = F.coalesce(F.col(target_col), F.lit(0)).alias(target_col)
    
    prepared_snowpark_df = sampled_snowpark_df.select(
        *feature_cols_filled,
        target_col_filled
    )

    # NOW convert to pandas only for train_test_split (necessary for sklearn)
    # This is the minimal conversion point - we've done all filtering/sampling in Snowpark
    prepared_pandas_df = prepared_snowpark_df.to_pandas()
    
    X = prepared_pandas_df[feature_cols_list]
    y = prepared_pandas_df[target_col]

    print(f"   Data shape: X={X.shape}, y={y.shape}")
    print(f"   Target range: [{y.min():.2f}, {y.max():.2f}], mean: {y.mean():.2f}")

    # Split into train and validation sets (requires pandas)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   Train: {X_train.shape[0]:,} samples, Val: {X_val.shape[0]:,} samples")

    # Prepare data for Tuner using DataConnector
    # DataConnector.from_dataframe() expects Snowpark DataFrames, not Pandas
    # Use explicit copy and reset index to avoid Pandas warnings
    train_data = X_train.reset_index(drop=True).copy()
    train_data[target_col] = np.asarray(y_train)

    val_data = X_val.reset_index(drop=True).copy()
    val_data[target_col] = np.asarray(y_val)

    train_snowpark = session.create_dataframe(train_data)
    val_snowpark = session.create_dataframe(val_data)

    train_dc = DataConnector.from_dataframe(train_snowpark)
    val_dc = DataConnector.from_dataframe(val_snowpark)

    # dataset_map must include both "train" and "test" keys (following HPO documentation)
    dataset_map = {"train": train_dc, "test": val_dc}

    # Model for this group (Snowflake ML class name)
    model_type = GROUP_MODEL.get(group_name, _DEFAULT_MODEL)
    search_space = SEARCH_SPACES.get(model_type, SEARCH_SPACES["XGBRegressor"])
    print(f"   Model: {model_type}")

    # Create training function for Tuner (model_type selects XGB/LGBM/SGD)
    train_func = create_train_func_for_tuner(feature_cols_list, model_type, target_col)

    tuner_config = TunerConfig(
        metric="rmse",
        mode="min",
        search_alg=BayesOpt(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0}),
        num_trials=NUM_TRIALS,
        max_concurrent_trials=MAX_CONCURRENT_TRIALS,
    )

    # Create and run Tuner
    print(f"   ⏳ Starting Bayesian Search ({model_type}, {NUM_TRIALS} trials)...")
    start_time = time.time()

    try:
        tuner = Tuner(train_func, search_space, tuner_config)
        results = tuner.run(dataset_map=dataset_map)

        elapsed_time = time.time() - start_time

        # Get best results (best_result is a single-row pd.DataFrame: config/* = hyperparams, other cols = metrics)
        best_result = results.best_result
        config_cols = [c for c in best_result.columns if str(c).startswith("config/")]

        def _to_native(v):
            if hasattr(v, "item"):
                return v.item()
            return v

        best_params = {
            str(c).replace("config/", ""): _to_native(best_result[c].iloc[0])
            for c in config_cols
        }
        best_rmse = float(best_result["rmse"].iloc[0]) if "rmse" in best_result.columns else None
        best_mae = float(best_result["mae"].iloc[0]) if "mae" in best_result.columns else None
        if best_rmse is None:
            raise ValueError("TunerResults best_result has no 'rmse' metric column")
        best_model = results.best_model

        # Evaluate best model on validation set again for consistency
        pred_result = best_model.predict(X_val)
        pred_df = pred_result.to_pandas() if hasattr(pred_result, "to_pandas") else pred_result
        out_col = best_model.get_output_cols()[0]
        y_val_pred = np.asarray(pred_df[out_col])

        # Validation metrics
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)

        abs_errors = np.abs(y_val - y_val_pred)
        denom_wape = np.sum(np.abs(y_val))
        val_wape = float(abs_errors.sum() / denom_wape) if denom_wape > 0 else 0.0

        non_zero_mask = np.abs(y_val) > 1e-8
        if non_zero_mask.any():
            val_mape = float(
                (np.abs(y_val[non_zero_mask] - y_val_pred[non_zero_mask]) / np.abs(y_val[non_zero_mask])).mean()
                * 100.0
            )
        else:
            val_mape = 0.0

        print(f"   ✅ Completed in {elapsed_time:.1f}s")
        print(f"      Best RMSE: {best_rmse:.4f}")
        print(
            f"      Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}, "
            f"WAPE: {val_wape:.4f}, MAPE: {val_mape:.2f}%"
        )

        # Store results in global dictionary (for summary later)
        all_results[group_name] = {
            "best_params": best_params,
            "best_cv_rmse": best_rmse,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_wape": val_wape,
            "val_mape": val_mape,
            "sample_size": sampled_count,
            "algorithm": model_type,
        }

        # Save to ML Experiments
        search_id = f"tune_{group_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiments_success = False

        if experiments_available:
            try:
                # Create a run for this group's best hyperparameters
                run_name = f"best_{group_name}_{datetime.now().strftime('%H%M%S')}"
                with exp_tracking.start_run(run_name):
                    # Log all hyperparameters
                    exp_tracking.log_params(best_params)

                    # Log metrics
                    exp_tracking.log_metrics(
                        {
                            "best_rmse": float(best_rmse),
                            "val_rmse": float(val_rmse),
                            "val_mae": float(val_mae),
                            "val_wape": float(val_wape),
                            "val_mape": float(val_mape),
                            "sample_size": int(sampled_count),
                            "num_trials": int(NUM_TRIALS),
                        }
                    )

                    # Log group identifier as a tag/parameter
                    exp_tracking.log_param("group_name", group_name)
                    exp_tracking.log_param("search_id", search_id)
                    exp_tracking.log_param("algorithm", model_type)
                    if feature_version_id:
                        exp_tracking.log_param("feature_version_id", feature_version_id)
                    if feature_snapshot_at is not None:
                        exp_tracking.log_param(
                            "feature_snapshot_at",
                            str(feature_snapshot_at),
                        )

                print(f"   ✅ Results logged to ML Experiments (run: {run_name})")
                experiments_success = True
            except Exception as e:
                print(f"   ⚠️  Error logging to Experiments: {str(e)[:200]}")
                experiments_success = False

        # Save to database ONLY if ML Experiments is not available or failed
        # If Experiments works, we don't need the table
        if not experiments_available or not experiments_success:
            print(
                f"   📋 Saving to table (Experiments {'not available' if not experiments_available else 'failed'})"
            )

            # Ensure table exists
            session.sql(
                f"""
                CREATE TABLE IF NOT EXISTS {HYPERPARAMETER_RESULTS_TABLE} (
                    search_id VARCHAR,
                    group_name VARCHAR,
                    algorithm VARCHAR,
                    best_params VARIANT,
                    best_cv_rmse FLOAT,
                    best_cv_mae FLOAT,
                    val_rmse FLOAT,
                    val_mae FLOAT,
                    n_iter INTEGER,
                    sample_size INTEGER,
                    feature_version_id VARCHAR,
                    feature_snapshot_at TIMESTAMP_NTZ,
                    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
                )
            """
            ).collect()
            # Ensure new columns exist if table already existed without them (one ALTER per column)
            for col_def in ["feature_version_id VARCHAR", "feature_snapshot_at TIMESTAMP_NTZ"]:
                try:
                    session.sql(
                        f"ALTER TABLE {HYPERPARAMETER_RESULTS_TABLE} ADD COLUMN IF NOT EXISTS {col_def}"
                    ).collect()
                except Exception:
                    pass

            best_params_json = json.dumps(
                {
                    k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in best_params.items()
                }
            )

            best_mae_value = best_mae if best_mae is not None else None
            best_mae_sql = (
                f"{best_mae_value:.6f}" if best_mae_value is not None else "NULL"
            )
            best_params_escaped = best_params_json.replace("'", "''")
            search_id_escaped = search_id.replace("'", "''")
            group_name_escaped = group_name.replace("'", "''")
            algorithm_escaped = model_type.replace("'", "''")
            if feature_version_id:
                feature_version_id_escaped = feature_version_id.replace("'", "''")
                feature_version_id_sql = f"'{feature_version_id_escaped}'"
            else:
                feature_version_id_sql = "NULL"
            if feature_snapshot_at is not None:
                feature_snapshot_at_sql = f"TO_TIMESTAMP_NTZ('{str(feature_snapshot_at)}')"
            else:
                feature_snapshot_at_sql = "NULL"

            insert_sql = f"""
                INSERT INTO {HYPERPARAMETER_RESULTS_TABLE}
                (search_id, group_name, algorithm, best_params, best_cv_rmse, best_cv_mae, val_rmse, val_mae, n_iter, sample_size, feature_version_id, feature_snapshot_at)
                VALUES (
                    '{search_id_escaped}',
                    '{group_name_escaped}',
                    '{algorithm_escaped}',
                    PARSE_JSON('{best_params_escaped}'),
                    {best_rmse:.6f},
                    {best_mae_sql},
                    {val_rmse:.6f},
                    {val_mae:.6f},
                    {NUM_TRIALS},
                    {sampled_count},
                    {feature_version_id_sql},
                    {feature_snapshot_at_sql}
                )
            """
            session.sql(insert_sql).collect()
            print(f"   ✅ Results saved to table")
        else:
            print(f"   ✅ Results stored in ML Experiments only (table not needed)")

        # Create result object to return
        class HyperparameterResult:
            def __init__(self):
                self.group_name = group_name
                self.best_params = best_params
                self.best_rmse = best_rmse
                self.val_rmse = val_rmse
                self.val_mae = val_mae
                self.skipped = False

        print(f"{'='*80}\n")
        return HyperparameterResult()

    except Exception as e:
        print(f"   ❌ Error during hyperparameter search: {str(e)[:200]}")
        import traceback

        print(f"   Traceback: {traceback.format_exc()[:300]}")
        print(f"   Will use default hyperparameters for this group.")
        print(f"{'='*80}\n")

        # Return a dummy object indicating failure
        return DummyResult()


# %% [markdown]
# ### 5c. Execute hyperparameter search (loop per group)

# %%
# Run hyperparameter search per group (loop; no MMT to avoid Ray serialization)
# Results are saved to BD_AA_DEV.SC_MODELS_BMX.HYPERPARAMETER_RESULTS and/or ML Experiments.
print("\n" + "=" * 80)
print("🚀 HYPERPARAMETER SEARCH PER GROUP (sequential loop)")
print("=" * 80)
print("\nRunning Bayesian Search with Tuner for each group (no MMT).\n")

start_time = time.time()
group_results = {}

stats_ntile_col = next((c for c in train_df.columns if c.upper() == STATS_NTILE_GROUP_COL), STATS_NTILE_GROUP_COL)
for idx, group_name in enumerate(groups_list, 1):
    print(f"\n[{idx}/{len(groups_list)}] Processing group: {group_name}")
    # Keep as Snowpark DataFrame - avoid to_pandas() here to prevent OOM
    group_snowpark = train_df.filter(train_df[stats_ntile_col] == group_name)
    # Pass Snowpark DataFrame directly - conversion to pandas happens inside function only when needed
    result = run_hyperparameter_search_for_one_group(group_name, group_snowpark)
    group_results[group_name] = result

end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60

print("\n" + "=" * 80)
print("✅ HYPERPARAMETER SEARCH COMPLETE")
print("=" * 80)
print(f"\n⏱️  Total search time: {elapsed_minutes:.2f} minutes")

# Review results by group
print("\n📊 Hyperparameter Search Results by Group:\n")
successful_searches = 0
for group_name in groups_list:
    result = group_results.get(group_name)
    if result is None:
        print(f"⚠️  {group_name}: No result")
        continue
    if getattr(result, "skipped", True):
        print(f"⚠️  {group_name}: Skipped (insufficient data or error)")
    else:
        print(f"✅ {group_name}:")
        print(f"   Best RMSE: {result.best_rmse:.4f}")
        print(f"   Val RMSE: {result.val_rmse:.4f}, Val MAE: {result.val_mae:.4f}")
        successful_searches += 1

print(
    f"\n✅ Completed hyperparameter search for {successful_searches}/{len(groups_list)} groups"
)

# %% [markdown]
# ## 5. Summary of All Results

# %%
print("\n" + "=" * 80)
print("📊 SUMMARY OF ALL HYPERPARAMETER SEARCHES")
print("=" * 80)

# Summary: Show results from Experiments or table
if experiments_available:
    print("\n📊 Results Summary:")
    print(f"   ✅ All results stored in ML Experiments")
    print(f"   ✅ Experiment: {experiment_name}")
    print(f"   ✅ Groups processed: {len(all_results)}")
    print(f"\n💡 View results in Snowsight: AI & ML → Experiments → {experiment_name}")

    # Try to show summary from Experiments if possible
    try:
        # This is a conceptual query - actual API may vary
        print("\n📊 Sample results from Experiments:")
        for group_name, result in list(all_results.items())[:5]:
            print(
                f"   {group_name}: RMSE={result['val_rmse']:.4f}, MAE={result['val_mae']:.4f}"
            )
        if len(all_results) > 5:
            print(f"   ... and {len(all_results) - 5} more groups")
    except:
        pass
else:
    # Fallback to table summary if Experiments not available
    print("\n📊 Results from Table:")
    summary_df = session.sql(
        f"""
        SELECT 
            group_name,
            best_cv_rmse,
            val_rmse,
            val_mae,
            sample_size,
            created_at
        FROM {HYPERPARAMETER_RESULTS_TABLE}
        WHERE created_at >= DATEADD(HOUR, -1, CURRENT_TIMESTAMP())
        ORDER BY group_name
    """
    )
    summary_df.show()

    # Overall statistics
    overall_stats = session.sql(
        f"""
        SELECT 
            COUNT(*) AS TOTAL_SEARCHES,
            AVG(best_cv_rmse) AS AVG_CV_RMSE,
            AVG(val_rmse) AS AVG_VAL_RMSE,
            MIN(val_rmse) AS MIN_VAL_RMSE,
            MAX(val_rmse) AS MAX_VAL_RMSE
        FROM {HYPERPARAMETER_RESULTS_TABLE}
        WHERE created_at >= DATEADD(HOUR, -1, CURRENT_TIMESTAMP())
    """
    )
    print("\n📊 Overall Statistics:")
    overall_stats.show()

# %% [markdown]
# ## 7. Scale Cluster Down

# %%
print("\n" + "=" * 80)
print("📉 SCALING CLUSTER DOWN")
print("=" * 80)

try:
    from snowflake.ml.runtime_cluster import scale_cluster

    print(f"⏳ Scaling cluster down to {CLUSTER_SIZE_DOWN} container...")
    scale_cluster(
        expected_cluster_size=CLUSTER_SIZE_DOWN
    )
    print(f"✅ Cluster scaled down to {CLUSTER_SIZE_DOWN} container")
except Exception as e:
    print(f"⚠️  Error scaling cluster down: {str(e)[:200]}")
    print("   Cluster may remain scaled...")

# %% [markdown]
# ## 8. Summary

# %%
print("\n" + "=" * 80)
print("✅ HYPERPARAMETER SEARCH COMPLETE!")
print("=" * 80)

print("\n📋 Summary:")
print(f"   ✅ Models: LGBMRegressor, XGBRegressor, SGDRegressor (per group)")
print(
    f"   ✅ Search method: Snowflake ML tune.search BayesOpt (per-group loop, no MMT)"
)
print(f"   ✅ Execution: Sequential loop over groups (avoids Ray serialization)")
print(f"   ✅ Groups processed: {successful_searches}/{len(groups_list)}")
print(f"   ✅ Trials per group: {NUM_TRIALS}")
print(f"   ✅ Sample rate per group: {SAMPLE_RATE_PER_GROUP*100:.0f}%")
print(f"   ⏱️  Total time: {elapsed_minutes:.2f} minutes")

# Calculate statistics from all_results if available
if all_results:
    avg_val_rmse = np.mean([r["val_rmse"] for r in all_results.values()])
    min_val_rmse = min([r["val_rmse"] for r in all_results.values()])
    max_val_rmse = max([r["val_rmse"] for r in all_results.values()])

    print(f"   ✅ Average Validation RMSE: {avg_val_rmse:.4f}")
    print(f"   ✅ Best Group RMSE: {min_val_rmse:.4f}")
    print(f"   ✅ Worst Group RMSE: {max_val_rmse:.4f}")

print("\n💡 Next Steps:")
print("   1. Review hyperparameter results by group")
print("   2. Run 04_many_model_training.py to train 16 models (one per group)")
print("   3. Each model will use its group-specific hyperparameters")

print("\n" + "=" * 80)
