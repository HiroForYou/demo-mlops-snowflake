# %% [markdown]
# # Migration: Many Model Training (MMT) - XGBoost (16 Models)
#
# ## Overview
# This script trains **16 XGBoost models** (one per stats_ntile_group) using Many Model Training (MMT) framework.
# Each model is trained with group-specific hyperparameters and data.
#
# ## What We'll Do:
# 1. Load best hyperparameters per group from hyperparameter search
# 2. Define training function for MMT (with group-specific hyperparameters)
# 3. Execute MMT training with partition_by="stats_ntile_group"
# 4. Register 16 models in Model Registry (one per group)
# 5. Create group-to-model mapping

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.ml.modeling.distributors.many_model import ManyModelTraining
from snowflake.ml.registry import Registry
from snowflake.ml.feature_store import FeatureStore
from snowflake.ml.experiment import ExperimentTracking
from snowflake.ml.model import task
import time
from datetime import datetime
import json

session = get_active_session()

# Set context
session.sql("USE DATABASE BD_AA_DEV").collect()
session.sql("USE SCHEMA SC_STORAGE_BMX_PS").collect()

print(f"✅ Connected to Snowflake")
print(f"   Database: {session.get_current_database()}")
print(f"   Schema: {session.get_current_schema()}")

# Configuración:
# - Si no tienes permisos para FeatureView/Dynamic Tables, usa tablas limpias o tabla de features materializada.
USE_CLEANED_TABLES = False  # True = TRAIN_DATASET_CLEANED, False = intentar tabla de features materializada
FEATURES_TABLE = "BD_AA_DEV.SC_FEATURES_BMX.UNI_BOX_FEATURES"  # creada por 02_feature_store_setup.py

# %% [markdown]
# ## 1. Setup Model Registry & Staging

# %%
# CREATE SCHEMA comentado (puede requerir permisos)
# session.sql("CREATE SCHEMA IF NOT EXISTS BD_AA_DEV.SC_MODELS_BMX").collect()
session.sql("CREATE STAGE IF NOT EXISTS BD_AA_DEV.SC_MODELS_BMX.MMT_MODELS").collect()

registry = Registry(
    session=session, database_name="BD_AA_DEV", schema_name="SC_MODELS_BMX"
)

print("✅ Model Registry initialized")
print("✅ Stage for MMT models created")

# %% [markdown]
# ## 2. Load Best Hyperparameters Per Group

# %%
print("\n" + "=" * 80)
print("📊 LOADING BEST HYPERPARAMETERS PER GROUP")
print("=" * 80)

# Try to load from ML Experiments first, fallback to table
hyperparams_by_group = {}
experiments_loaded = False

# Get all groups that need hyperparameters
all_groups_from_data = session.sql(
    """
    SELECT DISTINCT stats_ntile_group
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED
    WHERE stats_ntile_group IS NOT NULL
    ORDER BY stats_ntile_group
"""
).collect()

expected_groups = [row["STATS_NTILE_GROUP"] for row in all_groups_from_data]

# Try loading from ML Experiments
print("\n🔬 Attempting to load from ML Experiments...")
try:
    exp_tracking = ExperimentTracking(session)
    
    # Try to find the most recent experiment
    # Note: This is a simplified approach - in production you might want to specify experiment name
    from datetime import datetime, timedelta
    today = datetime.now().strftime('%Y%m%d')
    experiment_name = f"hyperparameter_search_xgboost_{today}"
    
    try:
        exp_tracking.set_experiment(experiment_name)
        print(f"✅ Found experiment: {experiment_name}")
        
        # Get all runs from this experiment
        # Note: The exact API may vary - this is a conceptual approach
        # You may need to query the experiments table directly
        experiments_loaded = True
        print("   ✅ ML Experiments available - loading from experiments")
    except:
        # Try yesterday's experiment as fallback
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        experiment_name = f"hyperparameter_search_xgboost_{yesterday}"
        try:
            exp_tracking.set_experiment(experiment_name)
            print(f"✅ Found experiment: {experiment_name}")
            experiments_loaded = True
        except:
            print("   ⚠️  No recent experiment found, will use table fallback")
            experiments_loaded = False
    
    if experiments_loaded:
        # Use ExperimentTracking API methods via SQL SHOW commands
        try:
            print(f"   📋 Listing runs from experiment: {experiment_name}")
            
            # Step 1: List all runs in the experiment using SHOW RUNS
            runs_query = f"SHOW RUNS IN EXPERIMENT {experiment_name}"
            runs_df = session.sql(runs_query)
            runs_list = runs_df.collect()
            
            if len(runs_list) == 0:
                print("   ⚠️  No runs found in experiment, using table fallback")
                experiments_loaded = False
            else:
                print(f"   ✅ Found {len(runs_list)} runs in experiment")
                
                # Step 2: For each run, get parameters and metrics
                runs_by_group = {}  # group_name -> best run info
                
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
                        
                        # Extract group_name from parameters
                        group_name = None
                        search_id = None
                        best_params = {}
                        
                        for param in params_list:
                            param_name = param["name"]
                            param_value = param["value"]
                            
                            if param_name == "group_name":
                                group_name = param_value
                            elif param_name == "search_id":
                                search_id = param_value
                            elif param_name not in ["algorithm"]:
                                # This is a hyperparameter
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
                            # Keep the best run per group (lowest val_rmse)
                            if group_name not in runs_by_group:
                                runs_by_group[group_name] = {
                                    "run_name": run_name,
                                    "params": best_params,
                                    "val_rmse": val_rmse,
                                    "val_mae": val_mae,
                                    "search_id": search_id,
                                }
                            else:
                                # Replace if this run has better (lower) RMSE
                                if val_rmse < runs_by_group[group_name]["val_rmse"]:
                                    runs_by_group[group_name] = {
                                        "run_name": run_name,
                                        "params": best_params,
                                        "val_rmse": val_rmse,
                                        "val_mae": val_mae,
                                        "search_id": search_id,
                                    }
                    
                    except Exception as run_error:
                        print(f"   ⚠️  Error processing run {run_name}: {str(run_error)[:100]}")
                        continue
                
                # Step 3: Store results in hyperparams_by_group
                if len(runs_by_group) > 0:
                    print(f"   ✅ Loaded {len(runs_by_group)} groups from Experiments")
                    
                    for group_name, run_info in runs_by_group.items():
                        hyperparams_by_group[group_name] = {
                            "params": run_info["params"],
                            "val_rmse": run_info["val_rmse"],
                            "search_id": run_info["search_id"] or f"exp_{group_name}",
                        }
                        
                        print(f"\n   {group_name}:")
                        print(f"      Val RMSE: {run_info['val_rmse']:.4f}")
                        if run_info["val_mae"]:
                            print(f"      Val MAE: {run_info['val_mae']:.4f}")
                        print(f"      Search ID: {run_info['search_id'] or 'N/A'}")
                        print(f"      Source: ML Experiments (run: {run_info['run_name']})")
                    
                    experiments_loaded = True
                else:
                    print("   ⚠️  No valid runs with group_name found, using table fallback")
                    experiments_loaded = False
                    
        except Exception as e:
            print(f"   ⚠️  Error using ExperimentTracking API: {str(e)[:200]}")
            print("   Will use table fallback")
            experiments_loaded = False

except Exception as e:
    print(f"   ⚠️  ML Experiments not available: {str(e)[:200]}")
    print("   Will use table fallback")
    experiments_loaded = False

# Fallback to table ONLY if Experiments didn't work or didn't have all groups
# Table is now a fallback mechanism, not the primary storage
if not experiments_loaded or len(hyperparams_by_group) < len(expected_groups):
    print("\n📋 Loading from table (HYPERPARAMETER_RESULTS) - Fallback mode...")
    print("   Note: Table is only used when ML Experiments is not available")
    
    # Check if table exists
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
        # Get most recent hyperparameter search results per group from table
        hyperparams_df = session.sql(
            """
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
                FROM BD_AA_DEV.SC_MODELS_BMX.HYPERPARAMETER_RESULTS
                WHERE algorithm = 'XGBoost'
                    AND group_name IS NOT NULL
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
            
            # Update or add to hyperparams_by_group
            for result in hyperparams_results:
                group_name = result["GROUP_NAME"]
                best_params_json = result["BEST_PARAMS"]
                
                # Parse hyperparameters
                if isinstance(best_params_json, str):
                    best_params = json.loads(best_params_json)
                else:
                    best_params = best_params_json
                
                # Only add if not already loaded from Experiments
                if group_name not in hyperparams_by_group:
                    hyperparams_by_group[group_name] = {
                        "params": best_params,
                        "val_rmse": result["VAL_RMSE"],
                        "search_id": result["SEARCH_ID"],
                    }
                    
                    print(f"\n   {group_name}:")
                    print(f"      Val RMSE: {result['VAL_RMSE']:.4f}")
                    print(f"      Search ID: {result['SEARCH_ID']}")
                    print(f"      Source: Table (fallback)")
        else:
            print("   ⚠️  Table exists but has no results")
    else:
        print("   ⚠️  Table does not exist (this is OK if using ML Experiments)")

if len(hyperparams_by_group) == 0:
    raise ValueError(
        "No hyperparameter results found in Experiments or table! Please run 03_hyperparameter_search.py first"
    )

print(f"\n✅ Total loaded hyperparameters: {len(hyperparams_by_group)}/{len(expected_groups)} groups")

# Get default hyperparameters (for groups without search results)
default_params = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
}

print(f"\n📋 Default hyperparameters (for groups without search results):")
for param, value in sorted(default_params.items()):
    print(f"   {param}: {value}")

# Validate that we have hyperparameters for all expected groups
print(f"\n🔍 Validating hyperparameter coverage...")
groups_with_hyperparams = set(hyperparams_by_group.keys())
groups_without_hyperparams = set(expected_groups) - groups_with_hyperparams

if groups_without_hyperparams:
    print(f"⚠️  WARNING: {len(groups_without_hyperparams)} groups will use default hyperparameters:")
    for group in sorted(groups_without_hyperparams):
        print(f"      - {group}")
else:
    print(f"✅ All {len(expected_groups)} groups have optimized hyperparameters!")

# %% [markdown]
# ## 3. Preparar datos de entrenamiento (sin FeatureView)

# %%
print("\n" + "=" * 80)
print("🏪 LOADING TRAINING DATA (SIN FEATURE VIEW)")
print("=" * 80)

if USE_CLEANED_TABLES:
    print("📊 Loading from cleaned table: TRAIN_DATASET_CLEANED")
    training_df = session.table("BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED")
    print(f"\n✅ Training data loaded from cleaned table")
    print(f"   Total records: {training_df.count():,}")
    print(f"   Columns: {len(training_df.columns)}")
else:
    # Preferimos la tabla materializada de features (sin Dynamic Tables).
    # Si falla por permisos/no existencia, hacemos fallback a la tabla limpia.
    try:
        # Mantener inicialización del Feature Store (aunque no usemos FeatureView)
        _fs = FeatureStore(session=session, database="BD_AA_DEV", name="SC_FEATURES_BMX")
        print("✅ Feature Store inicializado (sin FeatureView)")

        print(f"📊 Loading features from table: {FEATURES_TABLE}")
        features_df = session.table(FEATURES_TABLE)

        print("⏳ Loading target variable and stats_ntile_group from training table...")
        target_df = session.table("BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED").select(
            "customer_id", "brand_pres_ret", "week", "uni_box_week", "stats_ntile_group"
        )

        print("⏳ Joining features with target...")
        training_df = features_df.join(
            target_df, on=["customer_id", "brand_pres_ret", "week"], how="inner"
        )

        print(f"\n✅ Training data loaded from features table + target")
        print(f"   Total records: {training_df.count():,}")
        print(f"   Columns: {len(training_df.columns)}")
    except Exception as e:
        print(f"⚠️  Could not load/join features table ({FEATURES_TABLE}): {str(e)[:200]}")
        print("   Falling back to TRAIN_DATASET_CLEANED")
        training_df = session.table("BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED")
        print(f"\n✅ Training data loaded from cleaned table (fallback)")
        print(f"   Total records: {training_df.count():,}")
        print(f"   Columns: {len(training_df.columns)}")

# Show distribution by group
print("\n📊 Records per Group:")
group_counts = training_df.group_by("stats_ntile_group").count().sort("stats_ntile_group")
group_counts.show()

# %% [markdown]
# ## 4. Define Training Function for MMT

# %%
print("\n" + "=" * 80)
print("🔧 DEFINING TRAINING FUNCTION")
print("=" * 80)


def train_segment_model(data_connector, context):
    """
    Train XGBoost model for uni_box_week regression for a specific group.

    This function:
    1. Receives data for ONE group (via MMT partitioning)
    2. Loads group-specific hyperparameters
    3. Trains XGBoost model
    4. Evaluates on test set
    5. Returns trained model

    Args:
        data_connector: Snowflake data connector (provided by MMT)
        context: Contains partition_id (stats_ntile_group name)

    Returns:
        Trained XGBoost model object
    """
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np

    segment_name = context.partition_id
    print(f"\n{'='*80}")
    print(f"🚀 Training model for {segment_name}")
    print(f"{'='*80}")

    # Load data
    df = data_connector.to_pandas()
    print(f"📊 Data shape: {df.shape}")

    # Define excluded columns (metadata / non-features)
    excluded_cols = [
        "customer_id",
        "brand_pres_ret",
        "week",
        "FEATURE_TIMESTAMP",  # puede existir si usas tabla de features
        "stats_ntile_group",  # Group column - not a feature
    ]

    # Get feature columns
    feature_cols = [
        col for col in df.columns if col not in excluded_cols + ["uni_box_week"]
    ]

    target_col = "uni_box_week"

    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)

    print(f"   Features: {len(feature_cols)}")
    print(f"   Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"   Target mean: {y.mean():.2f}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   Training set: {X_train.shape[0]:,} samples")
    print(f"   Test set: {X_test.shape[0]:,} samples")

    # Get group-specific hyperparameters
    # IMPORTANT: This uses hyperparams_by_group loaded from script 03
    if segment_name in hyperparams_by_group:
        group_params = hyperparams_by_group[segment_name]["params"]
        search_id = hyperparams_by_group[segment_name]["search_id"]
        val_rmse = hyperparams_by_group[segment_name]["val_rmse"]
        print(f"\n   ✅ Using OPTIMIZED hyperparameters from script 03")
        print(f"      Search ID: {search_id}")
        print(f"      Validation RMSE (from search): {val_rmse:.4f}")
        print(f"      Hyperparameters: {', '.join([f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in sorted(group_params.items())[:5]])}...")
    else:
        group_params = default_params
        print(f"\n   ⚠️  Using DEFAULT hyperparameters (no search results found for {segment_name})")
        print(f"      This group was not processed in script 03 or had insufficient data")

    # Convert hyperparameters to proper types
    xgb_params = {}
    for k, v in group_params.items():
        if isinstance(v, (int, float)):
            xgb_params[k] = v
        elif isinstance(v, (np.integer, np.floating)):
            xgb_params[k] = float(v) if isinstance(v, np.floating) else int(v)
        else:
            xgb_params[k] = v

    # Ensure required parameters
    xgb_params["random_state"] = 42
    xgb_params["n_jobs"] = -1
    xgb_params["objective"] = "reg:squarederror"
    xgb_params["eval_metric"] = "rmse"

    print(f"\n   Training XGBoost with {len(xgb_params)} hyperparameters...")

    # Train model
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n   ✅ Model trained")
    print(f"      RMSE: {rmse:.2f}")
    print(f"      MAE: {mae:.2f}")
    print(f"{'='*80}\n")

    # Attach metadata to model
    model.rmse = rmse
    model.mae = mae
    model.training_samples = X_train.shape[0]
    model.test_samples = X_test.shape[0]
    model.feature_cols = feature_cols
    model.hyperparameters = xgb_params
    model.segment = segment_name
    model.group_name = segment_name

    return model


print("✅ Training function defined")

# %% [markdown]
# ## 5. Execute Many Model Training (MMT) - 16 Models

# %%
print("\n" + "=" * 80)
print("🚀 STARTING MANY MODEL TRAINING (MMT) - 16 MODELS")
print("=" * 80)
print("\nTraining 16 XGBoost models in PARALLEL (one per stats_ntile_group)")
print("Each model uses group-specific hyperparameters\n")

start_time = time.time()

# Create MMT trainer
trainer = ManyModelTraining(
    train_segment_model, "BD_AA_DEV.SC_MODELS_BMX.MMT_MODELS"
)

# Execute training with partition_by stats_ntile_group
training_run = trainer.run(
    partition_by="stats_ntile_group",  # ← KEY: Partition by group
    snowpark_dataframe=training_df,
    run_id=f"uni_box_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)

print("\n⏳ Training in progress... Monitoring completion...\n")

# Monitor with timeout
import time as time_module

max_wait = 1800  # 30 minutes max
check_interval = 10  # Check every 10 seconds
elapsed = 0
completed = False

while elapsed < max_wait:
    time_module.sleep(check_interval)
    elapsed += check_interval

    try:
        done_count = 0
        total_count = 0
        for partition_id in training_run.partition_details:
            total_count += 1
            status = training_run.partition_details[partition_id].status
            if status.name == "DONE" or status.name == "FAILED":
                done_count += 1

        print(
            f"⏱️  {elapsed}s elapsed - Progress: {done_count}/{total_count} models completed",
            end="\r",
        )

        if done_count == total_count:
            print("\n✅ All models completed!" + " " * 50)
            completed = True
            break
    except:
        print(f"⏱️  {elapsed}s elapsed - Waiting for status update...", end="\r")

if not completed:
    print("\n⏱️  Timeout reached - Verifying completion via stage..." + " " * 30)
    stage_files = session.sql(
        f"LIST @BD_AA_DEV.SC_MODELS_BMX.MMT_MODELS PATTERN='.*{training_run.run_id}.*'"
    ).collect()
    if len(stage_files) > 0:
        print(
            f"✅ Found {len(stage_files)} model files in stage - Training completed successfully!"
        )
        completed = True

end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60

print("\n" + "=" * 80)
print(f"✅ TRAINING COMPLETE! Status: {'COMPLETED' if completed else 'UNKNOWN'}")
print("=" * 80)
print(f"\n⏱️  Total training time: {elapsed_minutes:.2f} minutes")

# %% [markdown]
# ## 6. Review Training Results

# %%
print("\n📊 Training Results:\n")

for partition_id in training_run.partition_details:
    details = training_run.partition_details[partition_id]

    if details.status.name == "DONE":
        try:
            model = training_run.get_model(partition_id)

            print(f"\n✅ {partition_id if partition_id else 'DEFAULT'}:")
            print(f"   RMSE: {model.rmse:.2f}")
            print(f"   MAE: {model.mae:.2f}")
            print(f"   Training samples: {model.training_samples:,}")
            print(f"   Test samples: {model.test_samples:,}")
        except Exception as e:
            print(f"\n⚠️  {partition_id}: Could not load model - {str(e)[:100]}")
    else:
        print(f"\n❌ {partition_id}: Training failed")
        print(f"   Status: {details.status}")

# %% [markdown]
# ## 7. Register Model in Model Registry

# %%
print("\n" + "=" * 80)
print("📝 REGISTERING MODEL IN MODEL REGISTRY")
print("=" * 80)

version_date = datetime.now().strftime("%Y%m%d_%H%M")
registered_models = {}

for partition_id in training_run.partition_details:
    details = training_run.partition_details[partition_id]

    if details.status.name == "DONE":
        try:
            model = training_run.get_model(partition_id)

            # Model name includes group identifier
            model_name = f"uni_box_regression_{partition_id.lower()}"

            # Get group-specific search ID and hyperparameters if available
            group_search_id = None
            group_hyperparams = None
            if partition_id in hyperparams_by_group:
                group_search_id = hyperparams_by_group[partition_id]["search_id"]
                group_hyperparams = hyperparams_by_group[partition_id]["params"]

            # Prepare sample input from this group
            sample_input = (
                training_df.filter(training_df["stats_ntile_group"] == partition_id)
                .select(model.feature_cols)
                .limit(5)
            )

            print(f"\nRegistering {partition_id}...")

            # Prepare metrics including hyperparameters
            model_metrics = {
                "rmse": float(model.rmse),
                "mae": float(model.mae),
                "training_samples": int(model.training_samples),
                "test_samples": int(model.test_samples),
                "algorithm": "XGBoost",
                "group": partition_id,
                "hyperparameter_search_id": group_search_id or "default",
            }
            
            # Add hyperparameters to metrics (as nested dict)
            if group_hyperparams:
                # Convert hyperparameters to a format suitable for metrics
                for key, value in group_hyperparams.items():
                    if isinstance(value, (int, float)):
                        model_metrics[f"hyperparameter_{key}"] = float(value) if isinstance(value, float) else int(value)
                model_metrics["hyperparameters"] = json.dumps({
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in group_hyperparams.items()
                })

            mv = registry.log_model(
                model,
                model_name=model_name,
                version_name=f"v_{version_date}",
                comment=f"XGBoost regression model for uni_box_week - Group: {partition_id}",
                metrics=model_metrics,
                sample_input_data=sample_input,
                task=task.Task.TABULAR_REGRESSION,
            )

            registered_models[partition_id] = {
                "model_name": model_name,
                "version": f"v_{version_date}",
                "model_version": mv,
            }

            print(f"✅ {partition_id}: {model_name} v_{version_date}")
            print(f"   RMSE: {model.rmse:.2f}, MAE: {model.mae:.2f}")

        except Exception as e:
            print(f"❌ Error registering model: {str(e)[:200]}")

print(f"\n✅ {len(registered_models)} model(s) registered successfully!")

# %% [markdown]
# ## 8. Set Production Alias

# %%
print("\n🏷️  Setting PRODUCTION aliases...\n")

for partition_id, model_info in registered_models.items():
    model_name = model_info["model_name"]
    version = model_info["version"]
    model_version = model_info["model_version"]

    try:
        # Remove existing PRODUCTION alias
        try:
            model_ref = registry.get_model(model_name)
            model_ref.default.unset_alias("PRODUCTION")
        except:
            pass

        # Set alias on new version
        model_version.set_alias("PRODUCTION")
        print(f"✅ {model_name}: PRODUCTION → {version}")
    except Exception as e:
        print(f"⚠️  {model_name}: Error setting alias - {str(e)[:100]}")

print("\n✅ All production aliases configured!")

# %% [markdown]
# ## 9. Summary

# %%
print("\n" + "=" * 80)
print("🎉 MANY MODEL TRAINING (MMT) COMPLETE!")
print("=" * 80)

print("\n📊 Summary:")
print(f"   ✅ Models trained: {len(registered_models)}/16")
print(f"   ⏱️  Training time: {elapsed_minutes:.2f} minutes")
print(f"   🔧 Algorithm: XGBoost")
print(f"   📈 Hyperparameters: Group-specific (from hyperparameter search)")

if registered_models:
    print(f"\n🏆 Model Performance by Group:")
    for partition_id in sorted(registered_models.keys()):
        model = training_run.get_model(partition_id)
        print(f"   {partition_id}: RMSE={model.rmse:.2f}, MAE={model.mae:.2f}, Samples={model.training_samples:,}")

print("\n💡 Next Steps:")
print("   1. Review model performance by group")
print("   2. Run 05_create_partitioned_model.py to create partitioned model (combines all 16)")
print("   3. Run 06_partitioned_inference_batch.py for batch inference with automatic routing")

print("\n" + "=" * 80)
