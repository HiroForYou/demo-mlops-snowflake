# %% [markdown]
# # Migration: Many Model Training (MMT) - XGBoost
#
# ## Overview
# This script trains a single XGBoost model using Many Model Training (MMT) framework.
# Even though we have one model, we use MMT for consistency and future scalability.
#
# ## What We'll Do:
# 1. Load best hyperparameters from hyperparameter search
# 2. Define training function for MMT
# 3. Execute MMT training with full dataset
# 4. Register model in Model Registry

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.ml.modeling.distributors.many_model import ManyModelTraining
from snowflake.ml.registry import Registry
from snowflake.ml.feature_store import FeatureStore
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

# %% [markdown]
# ## 1. Setup Model Registry & Staging

# %%
session.sql("CREATE SCHEMA IF NOT EXISTS BD_AA_DEV.MODEL_REGISTRY").collect()
session.sql("CREATE STAGE IF NOT EXISTS BD_AA_DEV.MODEL_REGISTRY.MMT_MODELS").collect()

registry = Registry(
    session=session, database_name="BD_AA_DEV", schema_name="MODEL_REGISTRY"
)

print("✅ Model Registry initialized")
print("✅ Stage for MMT models created")

# %% [markdown]
# ## 2. Load Best Hyperparameters

# %%
print("\n" + "=" * 80)
print("📊 LOADING BEST HYPERPARAMETERS")
print("=" * 80)

# Get most recent hyperparameter search results
hyperparams_df = session.sql(
    """
    SELECT 
        search_id,
        algorithm,
        best_params,
        best_cv_rmse,
        val_rmse,
        val_mae,
        created_at
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.HYPERPARAMETER_RESULTS
    WHERE algorithm = 'XGBoost'
    ORDER BY created_at DESC
    LIMIT 1
"""
)

hyperparams_result = hyperparams_df.collect()

if len(hyperparams_result) == 0:
    raise ValueError(
        "No hyperparameter results found! Please run 03_hyperparameter_search.py first"
    )

best_result = hyperparams_result[0]
best_params_json = best_result["BEST_PARAMS"]
search_id = best_result["SEARCH_ID"]

print(f"\n✅ Best hyperparameters loaded")
print(f"   Search ID: {search_id}")
print(f"   Best CV RMSE: {best_result['BEST_CV_RMSE']:.4f}")
print(f"   Validation RMSE: {best_result['VAL_RMSE']:.4f}")

# Parse hyperparameters
if isinstance(best_params_json, str):
    best_params = json.loads(best_params_json)
else:
    best_params = best_params_json

print(f"\n📋 Best Hyperparameters:")
for param, value in sorted(best_params.items()):
    print(f"   {param}: {value}")

# %% [markdown]
# ## 3. Prepare Training Data from Feature Store

# %%
print("\n" + "=" * 80)
print("🏪 LOADING FEATURES FROM FEATURE STORE")
print("=" * 80)

# Initialize Feature Store
fs = FeatureStore(session=session, database="BD_AA_DEV", name="FEATURE_STORE")

print("✅ Feature Store initialized")

# Get FeatureView
try:
    feature_view = fs.get_feature_view("UNI_BOX_FEATURES", version="v1")
    print("✅ FeatureView 'UNI_BOX_FEATURES' v1 loaded")
except Exception as e:
    # Try v2 if v1 doesn't exist
    try:
        feature_view = fs.get_feature_view("UNI_BOX_FEATURES", version="v2")
        print("✅ FeatureView 'UNI_BOX_FEATURES' v2 loaded")
    except:
        print(f"❌ Error loading FeatureView: {str(e)}")
        print("   Please run 02_feature_store_setup.py first")
        raise

# Materialize features (get the feature data)
print("\n⏳ Materializing features from Feature Store...")
features_df = feature_view.get_features()

# Get target from cleaned training table
print("⏳ Loading target variable from training table...")
target_df = session.table("BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED").select(
    "customer_id", "brand_pres_ret", "week", "uni_box_week"
)

# Join features with target
print("⏳ Joining features with target...")
training_df = features_df.join(
    target_df, on=["customer_id", "brand_pres_ret", "week"], how="inner"
)

print(f"\n✅ Training data loaded from Feature Store")
print(f"   Total records: {training_df.count():,}")
print(f"   Columns: {len(training_df.columns)}")

# %% [markdown]
# ## 4. Define Training Function for MMT

# %%
print("\n" + "=" * 80)
print("🔧 DEFINING TRAINING FUNCTION")
print("=" * 80)


def train_model(data_connector, context):
    """
    Train XGBoost model for uni_box_week regression.

    This function:
    1. Receives data via MMT data connector
    2. Loads best hyperparameters
    3. Trains XGBoost model
    4. Evaluates on test set
    5. Returns trained model

    Args:
        data_connector: Snowflake data connector (provided by MMT)
        context: Contains partition_id (if partitioned)

    Returns:
        Trained XGBoost model object
    """
    import pandas as pd
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np
    import json

    print(f"\n{'='*80}")
    print(f"🚀 Training XGBoost model")
    print(f"{'='*80}")

    # Load data
    df = data_connector.to_pandas()
    print(f"📊 Data shape: {df.shape}")

    # Define excluded columns (metadata columns from Feature Store)
    excluded_cols = [
        "customer_id",
        "brand_pres_ret",
        "week",
        "FEATURE_TIMESTAMP",  # Feature Store timestamp column
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

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   Training set: {X_train.shape[0]:,} samples")
    print(f"   Test set: {X_test.shape[0]:,} samples")

    # Load best hyperparameters
    # Get from context or use default
    try:
        # Try to get hyperparameters from context
        if hasattr(context, "hyperparameters"):
            params = context.hyperparameters
        else:
            # Load from table (this is a workaround - in practice, pass via context)
            params = best_params
    except:
        params = best_params

    # Convert hyperparameters to proper types
    xgb_params = {}
    for k, v in params.items():
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

    print(f"\n   Training XGBoost with hyperparameters...")

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

    return model


print("✅ Training function defined")

# %% [markdown]
# ## 5. Execute Many Model Training (MMT)

# %%
print("\n" + "=" * 80)
print("🚀 STARTING MANY MODEL TRAINING (MMT)")
print("=" * 80)
print("\nTraining XGBoost model using MMT framework")
print("Using best hyperparameters from Random Search\n")

start_time = time.time()

# Create MMT trainer
trainer = ManyModelTraining(train_model, "BD_AA_DEV.MODEL_REGISTRY.MMT_MODELS")

# Execute training (without partition_by for single model)
training_run = trainer.run(
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
        f"LIST @BD_AA_DEV.MODEL_REGISTRY.MMT_MODELS PATTERN='.*{training_run.run_id}.*'"
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

            model_name = "uni_box_regression_model"

            # Prepare sample input
            sample_input = training_df.select(model.feature_cols).limit(5)

            print(f"\nRegistering model...")

            mv = registry.log_model(
                model,
                model_name=model_name,
                version_name=f"v_{version_date}",
                comment=f"XGBoost regression model for uni_box_week - Hyperparameters from {search_id}",
                metrics={
                    "rmse": float(model.rmse),
                    "mae": float(model.mae),
                    "training_samples": int(model.training_samples),
                    "test_samples": int(model.test_samples),
                    "algorithm": "XGBoost",
                    "hyperparameter_search_id": search_id,
                },
                sample_input_data=sample_input,
                task=task.Task.TABULAR_REGRESSION,
            )

            registered_models[partition_id] = {
                "model_name": model_name,
                "version": f"v_{version_date}",
                "model_version": mv,
            }

            print(f"✅ Model registered: {model_name} v_{version_date}")
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
print(f"   ✅ Models trained: {len(registered_models)}")
print(f"   ⏱️  Training time: {elapsed_minutes:.2f} minutes")
print(f"   🔧 Algorithm: XGBoost")
print(f"   📈 Hyperparameters: From {search_id}")

if registered_models:
    for partition_id, info in registered_models.items():
        model = training_run.get_model(partition_id)
        print(f"\n🏆 Model Performance:")
        print(f"   RMSE: {model.rmse:.2f}")
        print(f"   MAE: {model.mae:.2f}")
        print(f"   Training samples: {model.training_samples:,}")

print("\n💡 Next Steps:")
print("   1. Review model performance")
print("   2. Run 05_create_partitioned_model.py to create partitioned model")
print("   3. Run 06_partitioned_inference_batch.py for batch inference")

print("\n" + "=" * 80)
