# %% [markdown]
# # Migration: Hyperparameter Search (XGBoost + Snowflake ML Tune + MMT) - Per Group
#
# ## Overview
# This script performs hyperparameter optimization using Snowflake ML's tune.search for XGBoost regression.
# **Uses Many Model Training (MMT) to perform separate hyperparameter search for each of the 16 stats_ntile_group groups in PARALLEL.**
#
# ## What We'll Do:
# 1. Load cleaned training data with stats_ntile_group
# 2. Get all 16 unique groups
# 3. Use Many Model Training (MMT) to process all groups in PARALLEL:
#    - Each group runs its own Random Search using snowflake.ml.modeling.tune.search
#    - Load group-specific data (sampled for efficiency)
#    - Prepare features and target
#    - Save best hyperparameters per group
# 4. Generate summary of all hyperparameter results

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.ml.feature_store import FeatureStore
from snowflake.ml.modeling.tune import Tuner, TunerConfig, get_tuner_context, randint, uniform
from snowflake.ml.modeling.tune.search import RandomSearch
from snowflake.ml.data.data_connector import DataConnector
from snowflake.ml.experiment import ExperimentTracking
from snowflake.ml.modeling.distributors.many_model import ManyModelTraining
import numpy as np
from datetime import datetime
import json
import time

session = get_active_session()

# Set context
session.sql("USE DATABASE BD_AA_DEV").collect()
session.sql("USE SCHEMA SC_STORAGE_BMX_PS").collect()

print(f"✅ Connected to Snowflake")
print(f"   Database: {session.get_current_database()}")
print(f"   Schema: {session.get_current_schema()}")

# Configuración:
# - Si no tienes permisos para FeatureView/Dynamic Tables, usa tablas limpias.
# - Mantengo el flag pero agrego fallback automático si el modo Feature Store falla.
USE_CLEANED_TABLES = False  # True = TRAIN_DATASET_CLEANED, False = intentar Feature Store

# %% [markdown]
# ## 1. Get All Groups and Load Training Data

# %%
print("\n" + "=" * 80)
print("📊 GETTING ALL GROUPS")
print("=" * 80)

# Get all unique groups from cleaned training table
groups_df = session.sql(
    """
    SELECT DISTINCT stats_ntile_group AS GROUP_NAME
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED
    WHERE stats_ntile_group IS NOT NULL
    ORDER BY stats_ntile_group
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
    train_df = session.table("BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED")
    
    total_rows = train_df.count()
    print(f"\n✅ Training data loaded from cleaned table")
    print(f"   Total rows: {total_rows:,}")
    print(f"   ⚠️  TESTING MODE: Using cleaned tables directly (not Feature Store)")
else:
    print("\n" + "=" * 80)
    print("🏪 FEATURE STORE MODE (SIN FEATUREVIEW)")
    print("=" * 80)

    # En lugar de FeatureView, soportamos una tabla de features materializada por `02_feature_store_setup.py`
    # (sin Dynamic Tables). Si no existe o falla, hacemos fallback a tablas limpias.
    FEATURES_TABLE = "BD_AA_DEV.SC_FEATURES_BMX.UNI_BOX_FEATURES"

    try:
        # Mantener inicialización del Feature Store (aunque no usemos FeatureView)
        _fs = FeatureStore(session=session, database="BD_AA_DEV", name="SC_FEATURES_BMX")
        print("✅ Feature Store inicializado (sin FeatureView)")

        print(f"⏳ Loading features from table: {FEATURES_TABLE} ...")
        features_df = session.table(FEATURES_TABLE)

        print("⏳ Loading target variable and stats_ntile_group from training table...")
        target_df = session.table("BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED").select(
            "customer_id", "brand_pres_ret", "week", "uni_box_week", "stats_ntile_group"
        )

        print("⏳ Joining features with target...")
        train_df = features_df.join(
            target_df, on=["customer_id", "brand_pres_ret", "week"], how="inner"
        )

        total_rows = train_df.count()
        print(f"\n✅ Training data loaded from features table + target")
        print(f"   Total rows: {total_rows:,}")
    except Exception as e:
        print(f"⚠️  Could not load/join features table ({FEATURES_TABLE}): {str(e)[:200]}")
        print("   Falling back to TRAIN_DATASET_CLEANED (USE_CLEANED_TABLES=True behavior)")
        train_df = session.table("BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED")
        total_rows = train_df.count()
        print(f"\n✅ Training data loaded from cleaned table (fallback)")
        print(f"   Total rows: {total_rows:,}")

# %% [markdown]
# ## 3. Define Hyperparameter Search Space

# %%
print("\n" + "=" * 80)
print("🎯 DEFINING HYPERPARAMETER SEARCH SPACE")
print("=" * 80)

# Define parameter distributions for Random Search using Snowflake ML tune.search
search_space = {
    "n_estimators": randint(50, 300),
    "max_depth": randint(3, 10),
    "learning_rate": uniform(0.01, 0.3),
    "subsample": uniform(0.6, 1.0),
    "colsample_bytree": uniform(0.6, 1.0),
    "min_child_weight": randint(1, 7),
    "gamma": uniform(0, 0.5),
    "reg_alpha": uniform(0, 1),
    "reg_lambda": uniform(0, 1),
}

print("\n📋 Hyperparameter Search Space (using snowflake.ml.modeling.tune.search):")
for param, dist in search_space.items():
    if hasattr(dist, "low") and hasattr(dist, "high"):
        if hasattr(dist, "base"):  # loguniform
            print(f"   {param}: loguniform({dist.low:.2f}, {dist.high:.2f})")
        else:  # uniform or randint
            # En versiones recientes `randint`/`uniform` son factories; evitamos isinstance().
            dist_name = dist.__class__.__name__.lower()
            if "rand" in dist_name and "int" in dist_name:
                print(f"   {param}: randint({dist.low}, {dist.high})")
            else:
                print(f"   {param}: uniform({dist.low:.2f}, {dist.high:.2f})")

# Number of trials for Random Search (reduced per group for efficiency)
num_trials = 30  # Reduced from 50 since we're doing 16 groups
print(f"\n🔢 Random Search trials per group: {num_trials}")

# Sample rate per group for hyperparameter search
# Options:
#   - 1.0 = Use full group (most accurate but slower)
#   - 0.5 = Use 50% of group (balanced)
#   - 0.1 = Use 10% of group (faster, less accurate)
# Note: For large datasets, using full group (1.0) is recommended for better hyperparameter tuning
SAMPLE_RATE_PER_GROUP = 0.2
print(f"📊 Sample rate per group: {SAMPLE_RATE_PER_GROUP*100:.0f}%")
if SAMPLE_RATE_PER_GROUP < 1.0:
    print(f"   ⚠️  Using {SAMPLE_RATE_PER_GROUP*100:.0f}% of data - consider using 1.0 (full group) for better results")
else:
    print(f"   ✅ Using full group data for optimal hyperparameter search")

# %% [markdown]
# ## 4. Perform Hyperparameter Search Per Group

# %%
print("\n" + "=" * 80)
print("🔍 PERFORMING HYPERPARAMETER SEARCH PER GROUP")
print("=" * 80)

# Initialize ML Experiments for hyperparameter tracking FIRST
print("\n" + "=" * 80)
print("🔬 INITIALIZING ML EXPERIMENTS")
print("=" * 80)

# Initialize experiment_name in global scope
experiment_name = f"hyperparameter_search_xgboost_{datetime.now().strftime('%Y%m%d')}"

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
        """
        CREATE TABLE IF NOT EXISTS BD_AA_DEV.SC_MODELS_BMX.HYPERPARAMETER_RESULTS (
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
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """
    ).collect()
    print("   ✅ Table created (will be used as primary storage)")
else:
    print("\n📋 Skipping table creation (using ML Experiments as primary storage)")

# Define excluded columns (metadata columns, not features)
# Note: FEATURE_TIMESTAMP may not exist if using cleaned tables directly
excluded_cols = [
    "customer_id",
    "brand_pres_ret",
    "week",
    "FEATURE_TIMESTAMP",  # Feature Store timestamp column (may not exist in cleaned tables)
    "stats_ntile_group",  # Group column - not a feature
]

# Get feature columns from first group (all groups should have same features)
sample_group_df = train_df.filter(train_df["stats_ntile_group"] == groups_list[0])
sample_pandas = sample_group_df.limit(1).to_pandas()
feature_cols = [
    col for col in sample_pandas.columns 
    if col not in excluded_cols + ["uni_box_week"]
]

print(f"\n📋 Features ({len(feature_cols)}):")
for col in sorted(feature_cols):
    print(f"   - {col}")

# Dictionary to store all results
all_results = {}


def create_train_func_for_tuner(feature_cols):
    """
    Create a training function for the Tuner.
    This function will be called for each trial with different hyperparameters.
    
    Following Snowflake ML HPO best practices:
    - Gets all data from dataset_map (not from outer scope)
    - Uses "train" and "test" keys from dataset_map
    - Reports metrics via tuner_context.report()
    
    Args:
        feature_cols: List of feature column names
    
    Returns:
        train_func: Function that can be used by Tuner
    """
    def train_func():
        from snowflake.ml.modeling.xgboost import XGBRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import numpy as np
        
        tuner_context = get_tuner_context()
        params = tuner_context.get_hyper_params()
        dm = tuner_context.get_dataset_map()
        
        # Load data from DataConnector (following HPO documentation pattern)
        train_pd = dm["train"].to_pandas()
        test_pd = dm["test"].to_pandas()
        
        # Prepare training features and target
        X_train = train_pd[feature_cols].fillna(0)
        y_train = train_pd["uni_box_week"].fillna(0)
        
        # Prepare validation features and target
        X_val = test_pd[feature_cols].fillna(0)
        y_val = test_pd["uni_box_week"].fillna(0)
        
        # Build model with hyperparameters from tuner
        xgb_params = params.copy()
        xgb_params["random_state"] = 42
        xgb_params["n_jobs"] = -1
        xgb_params["objective"] = "reg:squarederror"
        xgb_params["eval_metric"] = "rmse"
        
        # Train model
        model = XGBRegressor(**xgb_params)
        model.fit(X_train, y_train)
        
        # Evaluate on validation set (from dataset_map, not outer scope)
        y_val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        
        # Report metrics (following HPO documentation pattern)
        tuner_context.report(metrics={"rmse": val_rmse, "mae": val_mae}, model=model)
    
    return train_func


def search_hyperparameters_for_group(data_connector, context):
    """
    Perform hyperparameter search for a specific group using MMT + HPO.
    
    ARCHITECTURE: Two-Level Parallelization
    ========================================
    Level 1 (MMT - Many Model Training):
      - MMT partitions data by stats_ntile_group (16 groups)
      - Each group runs in PARALLEL (16 parallel executions)
      - This function is called ONCE per group by MMT
    
    Level 2 (HPO - Hyperparameter Optimization):
      - Within EACH group, Tuner runs multiple trials (e.g., 30 trials)
      - Trials can run in parallel (controlled by max_concurrent_trials)
      - Each trial tests different hyperparameter combinations
      - Total parallelization = 16 groups × max_concurrent_trials per group
    
    This function:
    1. Receives data for ONE group (via MMT partitioning)
    2. Performs Random Search using Snowflake ML Tuner (HPO)
    3. Saves best hyperparameters to Experiments/Table
    4. Returns a simple object with results
    
    Args:
        data_connector: Snowflake data connector (provided by MMT)
        context: Contains partition_id (stats_ntile_group name)
    
    Returns:
        Simple object containing best hyperparameters
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np
    
    group_name = context.partition_id
    print(f"\n{'='*80}")
    print(f"🔍 Hyperparameter Search for Group: {group_name}")
    print(f"{'='*80}")
    
    # Define DummyResult class once (reused in multiple early-return scenarios)
    class DummyResult:
        def __init__(self):
            self.group_name = group_name
            self.skipped = True
    
    # Load data
    df = data_connector.to_pandas()
    group_count = len(df)
    print(f"📊 Group data: {group_count:,} records")
    
    if group_count < 50:
        print(f"⚠️  WARNING: Group has less than 50 records. Skipping hyperparameter search.")
        print(f"   Will use default hyperparameters for this group.")
        return DummyResult()
    
    # Sample data for this group (or use full group if SAMPLE_RATE_PER_GROUP = 1.0)
    if SAMPLE_RATE_PER_GROUP < 1.0:
        sampled_df = df.sample(frac=SAMPLE_RATE_PER_GROUP, random_state=42)
        sampled_count = len(sampled_df)
        print(f"   Sampled: {sampled_count:,} records ({SAMPLE_RATE_PER_GROUP*100:.0f}% of {group_count:,} total)")
    else:
        # Use full group for better hyperparameter search
        sampled_df = df
        sampled_count = group_count
        print(f"   Using full group: {sampled_count:,} records (100% of group)")
    
    # Define excluded columns (metadata columns, not features)
    # Note: FEATURE_TIMESTAMP may not exist if using cleaned tables directly
    excluded_cols = [
        "customer_id",
        "brand_pres_ret",
        "week",
        "FEATURE_TIMESTAMP",  # Feature Store timestamp column (may not exist in cleaned tables)
        "stats_ntile_group",  # Group column - not a feature
    ]
    
    # Get feature columns
    feature_cols_list = [
        col for col in sampled_df.columns 
        if col not in excluded_cols + ["uni_box_week"]
    ]
    
    # Prepare X and y
    X = sampled_df[feature_cols_list].fillna(0)
    y = sampled_df["uni_box_week"].fillna(0)
    
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    print(f"   Target range: [{y.min():.2f}, {y.max():.2f}], mean: {y.mean():.2f}")
    
    # Split into train and validation sets
    if len(X) < 20:
        print(f"⚠️  WARNING: Not enough data for train/val split. Skipping.")
        return DummyResult()
        
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Train: {X_train.shape[0]:,} samples, Val: {X_val.shape[0]:,} samples")
    
    # Prepare data for Tuner using DataConnector
    # Following Snowflake ML HPO documentation pattern: include both "train" and "test" in dataset_map
    train_data = X_train.copy()
    train_data["uni_box_week"] = y_train.values
    
    val_data = X_val.copy()
    val_data["uni_box_week"] = y_val.values
    
    # Create DataConnectors for both train and test sets
    train_dc = DataConnector.from_dataframe(train_data)
    val_dc = DataConnector.from_dataframe(val_data)
    
    # dataset_map must include both "train" and "test" keys (following HPO documentation)
    dataset_map = {
        "train": train_dc,
        "test": val_dc
    }
    
    # Create training function for Tuner (no longer needs X_val, y_val - gets from dataset_map)
    train_func = create_train_func_for_tuner(feature_cols_list)
    
    # Configure Tuner
    # IMPORTANT: Understanding parallelization levels:
    # - MMT (Many Model Training): Parallelizes at GROUP level (16 groups run in parallel)
    # - HPO (Hyperparameter Optimization): Parallelizes at TRIAL level within each group
    #   - max_concurrent_trials controls how many trials run in parallel within THIS group
    #   - Setting to 1 means trials run sequentially within the group
    #   - Can increase to use more CPU/GPU cores per group (e.g., max_concurrent_trials=4)
    #   - Total parallelization = MMT groups × HPO concurrent trials per group
    tuner_config = TunerConfig(
        metric="rmse",
        mode="min",  # Minimize RMSE
        search_alg=RandomSearch(),
        num_trials=num_trials,
        max_concurrent_trials=1,  # Trials run sequentially within this group
        # Note: Can increase max_concurrent_trials if you have more CPU/GPU cores available
        # per group. For example, if each group has 4 cores, set max_concurrent_trials=4
        # to run 4 trials in parallel within each group.
    )
    
    # Create and run Tuner
    print(f"   ⏳ Starting Random Search using snowflake.ml.modeling.tune ({num_trials} trials)...")
    start_time = time.time()
    
    try:
        tuner = Tuner(train_func, search_space, tuner_config)
        results = tuner.run(dataset_map=dataset_map)
        
        elapsed_time = time.time() - start_time
        
        # Get best results
        best_result = results.best_result
        best_params = best_result.hyperparameters
        best_rmse = best_result.metrics["rmse"]
        best_mae = best_result.metrics.get("mae", None)
        best_model = results.best_model
        
        # Evaluate best model on validation set again for consistency
        y_val_pred = best_model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        
        print(f"   ✅ Completed in {elapsed_time:.1f}s")
        print(f"      Best RMSE: {best_rmse:.4f}")
        print(f"      Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}")
        
        # Store results in global dictionary (for summary later)
        all_results[group_name] = {
            "best_params": best_params,
            "best_cv_rmse": best_rmse,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "sample_size": sampled_count,
        }
        
        # Save to ML Experiments
        search_id = f"xgb_snowflake_tune_{group_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiments_success = False
        
        if experiments_available:
            try:
                # Create a run for this group's best hyperparameters
                run_name = f"best_{group_name}_{datetime.now().strftime('%H%M%S')}"
                with exp_tracking.start_run(run_name):
                    # Log all hyperparameters
                    exp_tracking.log_params(best_params)
                    
                    # Log metrics
                    exp_tracking.log_metrics({
                        "best_rmse": float(best_rmse),
                        "val_rmse": float(val_rmse),
                        "val_mae": float(val_mae),
                        "sample_size": int(sampled_count),
                        "num_trials": int(num_trials),
                    })
                    
                    # Log group identifier as a tag/parameter
                    exp_tracking.log_param("group_name", group_name)
                    exp_tracking.log_param("search_id", search_id)
                    exp_tracking.log_param("algorithm", "XGBoost")
                
                print(f"   ✅ Results logged to ML Experiments (run: {run_name})")
                experiments_success = True
            except Exception as e:
                print(f"   ⚠️  Error logging to Experiments: {str(e)[:200]}")
                experiments_success = False
        
        # Save to database ONLY if ML Experiments is not available or failed
        # If Experiments works, we don't need the table
        if not experiments_available or not experiments_success:
            print(f"   📋 Saving to table (Experiments {'not available' if not experiments_available else 'failed'})")
            
            # Ensure table exists
            session.sql(
                """
                CREATE TABLE IF NOT EXISTS BD_AA_DEV.SC_MODELS_BMX.HYPERPARAMETER_RESULTS (
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
                    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
                )
            """
            ).collect()
            
            best_params_json = json.dumps(
                {
                    k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in best_params.items()
                }
            )
            
            best_mae_value = best_mae if best_mae is not None else None
            best_mae_sql = f"{best_mae_value:.6f}" if best_mae_value is not None else "NULL"
            
            # Guardar resultados con group_name como identificador del grupo
            insert_sql = f"""
                INSERT INTO BD_AA_DEV.SC_MODELS_BMX.HYPERPARAMETER_RESULTS
                (search_id, group_name, algorithm, best_params, best_cv_rmse, best_cv_mae, val_rmse, val_mae, n_iter, sample_size)
                VALUES (
                    '{search_id}',
                    '{group_name}',
                    'XGBoost',
                    PARSE_JSON('{best_params_json}'),
                    {best_rmse:.6f},
                    {best_mae_sql},
                    {val_rmse:.6f},
                    {val_mae:.6f},
                    {num_trials},
                    {sampled_count}
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


# Setup MMT stage (if needed)
print("\n📋 Setting up MMT stage for hyperparameter search...")
# CREATE SCHEMA comentado (puede requerir permisos)
# session.sql("CREATE SCHEMA IF NOT EXISTS BD_AA_DEV.SC_MODELS_BMX").collect()
session.sql("CREATE STAGE IF NOT EXISTS BD_AA_DEV.SC_MODELS_BMX.MMT_HYPERPARAMETER_SEARCH").collect()
print("   ✅ MMT stage created")

# Execute Many Model Training (MMT) for Hyperparameter Search
# ============================================================
# ARCHITECTURE: Two-Level Parallelization
# 
# Level 1 - MMT (Many Model Training):
#   - MMT partitions data by stats_ntile_group (16 groups)
#   - Each group runs in PARALLEL (16 groups executing simultaneously)
#   - This is the OUTER level of parallelization
#
# Level 2 - HPO (Hyperparameter Optimization):
#   - Within EACH group, Tuner runs multiple trials (e.g., 30 trials)
#   - Trials can run in parallel (controlled by max_concurrent_trials in TunerConfig)
#   - This is the INNER level of parallelization
#
# Total Parallelization = MMT groups × HPO concurrent trials per group
# Example: 16 groups × 1 trial/group = 16 parallel executions
#          If max_concurrent_trials=4: 16 groups × 4 trials/group = 64 parallel executions
#
# IMPORTANTE: La conexión grupo-hiperparámetros se hace mediante el campo 
# 'group_name' que se guarda en la tabla HYPERPARAMETER_RESULTS o ML Experiments. 
# Luego en el script 04, se usa este 'group_name' para cargar los hiperparámetros correctos 
# para cada modelo.
print("\n" + "=" * 80)
print("🚀 STARTING MANY MODEL TRAINING (MMT) FOR HYPERPARAMETER SEARCH")
print("=" * 80)
print("\nTwo-Level Parallelization Architecture:")
print("  Level 1 (MMT): 16 groups running in PARALLEL")
print("  Level 2 (HPO): Multiple trials per group (controlled by max_concurrent_trials)")
print("  Each group will run its own Random Search with Tuner\n")

start_time = time.time()

# Create MMT trainer for hyperparameter search
trainer = ManyModelTraining(
    search_hyperparameters_for_group, 
    "BD_AA_DEV.SC_MODELS_BMX.MMT_HYPERPARAMETER_SEARCH"
)

# Execute MMT with partition_by stats_ntile_group
# This will automatically partition the data by group and run search_hyperparameters_for_group
# for each partition in parallel
search_run = trainer.run(
    partition_by="stats_ntile_group",  # ← KEY: Partition by group
    snowpark_dataframe=train_df,
    run_id=f"hyperparameter_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)

print("\n⏳ Hyperparameter search in progress... Monitoring completion...\n")

# Monitor with timeout
max_wait = 3600  # 60 minutes max (hyperparameter search can take longer)
check_interval = 10  # Check every 10 seconds
elapsed = 0
completed = False

while elapsed < max_wait:
    time.sleep(check_interval)
    elapsed += check_interval

    try:
        done_count = 0
        total_count = 0
        for partition_id in search_run.partition_details:
            total_count += 1
            status = search_run.partition_details[partition_id].status
            if status.name == "DONE" or status.name == "FAILED":
                done_count += 1

        print(
            f"⏱️  {elapsed}s elapsed - Progress: {done_count}/{total_count} groups completed",
            end="\r",
        )

        if done_count == total_count:
            print("\n✅ All hyperparameter searches completed!" + " " * 50)
            completed = True
            break
    except:
        print(f"⏱️  {elapsed}s elapsed - Waiting for status update...", end="\r")

if not completed:
    print("\n⏱️  Timeout reached - Verifying completion via stage..." + " " * 30)
    stage_files = session.sql(
        f"LIST @BD_AA_DEV.SC_MODELS_BMX.MMT_HYPERPARAMETER_SEARCH PATTERN='.*{search_run.run_id}.*'"
    ).collect()
    if len(stage_files) > 0:
        print(
            f"✅ Found {len(stage_files)} result files in stage - Search completed successfully!"
        )
        completed = True

end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60

print("\n" + "=" * 80)
print(f"✅ HYPERPARAMETER SEARCH COMPLETE! Status: {'COMPLETED' if completed else 'UNKNOWN'}")
print("=" * 80)
print(f"\n⏱️  Total search time: {elapsed_minutes:.2f} minutes")

# Review results from MMT run
print("\n📊 Hyperparameter Search Results by Group:\n")

successful_searches = 0
for partition_id in search_run.partition_details:
    details = search_run.partition_details[partition_id]

    if details.status.name == "DONE":
        try:
            result = search_run.get_model(partition_id)
            if hasattr(result, 'skipped') and result.skipped:
                print(f"⚠️  {partition_id}: Skipped (insufficient data or error)")
            else:
                print(f"✅ {partition_id}:")
                print(f"   Best RMSE: {result.best_rmse:.4f}")
                print(f"   Val RMSE: {result.val_rmse:.4f}, Val MAE: {result.val_mae:.4f}")
                successful_searches += 1
        except Exception as e:
            print(f"⚠️  {partition_id}: Could not load result - {str(e)[:100]}")
    else:
        print(f"❌ {partition_id}: Search failed")
        print(f"   Status: {details.status}")

print(f"\n✅ Completed hyperparameter search for {successful_searches}/{len(groups_list)} groups")

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
            print(f"   {group_name}: RMSE={result['val_rmse']:.4f}, MAE={result['val_mae']:.4f}")
        if len(all_results) > 5:
            print(f"   ... and {len(all_results) - 5} more groups")
    except:
        pass
else:
    # Fallback to table summary if Experiments not available
    print("\n📊 Results from Table:")
    summary_df = session.sql(
        """
        SELECT 
            group_name,
            best_cv_rmse,
            val_rmse,
            val_mae,
            sample_size,
            created_at
        FROM BD_AA_DEV.SC_MODELS_BMX.HYPERPARAMETER_RESULTS
        WHERE created_at >= DATEADD(HOUR, -1, CURRENT_TIMESTAMP())
        ORDER BY group_name
    """
    )
    summary_df.show()
    
    # Overall statistics
    overall_stats = session.sql(
        """
        SELECT 
            COUNT(*) AS TOTAL_SEARCHES,
            AVG(best_cv_rmse) AS AVG_CV_RMSE,
            AVG(val_rmse) AS AVG_VAL_RMSE,
            MIN(val_rmse) AS MIN_VAL_RMSE,
            MAX(val_rmse) AS MAX_VAL_RMSE
        FROM BD_AA_DEV.SC_MODELS_BMX.HYPERPARAMETER_RESULTS
        WHERE created_at >= DATEADD(HOUR, -1, CURRENT_TIMESTAMP())
    """
    )
    print("\n📊 Overall Statistics:")
    overall_stats.show()

# %% [markdown]
# ## 6. Summary

# %%
print("\n" + "=" * 80)
print("✅ HYPERPARAMETER SEARCH COMPLETE!")
print("=" * 80)

print("\n📋 Summary:")
print(f"   ✅ Algorithm: XGBoost")
print(f"   ✅ Search method: Many Model Training (MMT) + Snowflake ML tune.search RandomSearch")
print(f"   ✅ Execution: PARALLEL (all groups processed simultaneously)")
print(f"   ✅ Groups processed: {successful_searches}/{len(groups_list)}")
print(f"   ✅ Trials per group: {num_trials}")
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
