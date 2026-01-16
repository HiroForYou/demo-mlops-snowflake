# %% [markdown]
# # Migration: Hyperparameter Search (XGBoost + Snowflake ML Tune) - Per Group
#
# ## Overview
# This script performs hyperparameter optimization using Snowflake ML's tune.search for XGBoost regression.
# **Performs separate hyperparameter search for each of the 16 stats_ntile_group groups.**
#
# ## What We'll Do:
# 1. Load cleaned training data with stats_ntile_group
# 2. Get all 16 unique groups
# 3. For each group:
#    - Load group-specific data (sampled for efficiency)
#    - Prepare features and target
#    - Perform Random Search using snowflake.ml.modeling.tune.search
#    - Save best hyperparameters per group
# 4. Generate summary of all hyperparameter results

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.ml.feature_store import FeatureStore
from snowflake.ml.modeling.tune import Tuner, TunerConfig, get_tuner_context
from snowflake.ml.modeling.tune.search import RandomSearch, randint, uniform
from snowflake.ml.data.data_connector import DataConnector
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
# ## 2. Load Features from Feature Store

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

# Get target and stats_ntile_group from cleaned training table
print("⏳ Loading target variable and stats_ntile_group from training table...")
target_df = session.table("BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED").select(
    "customer_id", "brand_pres_ret", "week", "uni_box_week", "stats_ntile_group"
)

# Join features with target
print("⏳ Joining features with target...")
train_df = features_df.join(
    target_df, on=["customer_id", "brand_pres_ret", "week"], how="inner"
)

total_rows = train_df.count()
print(f"\n✅ Training data loaded from Feature Store")
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
            if isinstance(dist, randint):
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
sample_rate = 0.1
print(f"📊 Sample rate per group: {sample_rate*100:.0f}%")
if sample_rate < 1.0:
    print(f"   ⚠️  Using {sample_rate*100:.0f}% of data - consider using 1.0 (full group) for better results")
else:
    print(f"   ✅ Using full group data for optimal hyperparameter search")

# %% [markdown]
# ## 4. Perform Hyperparameter Search Per Group

# %%
print("\n" + "=" * 80)
print("🔍 PERFORMING HYPERPARAMETER SEARCH PER GROUP")
print("=" * 80)

# Create results table (with group column)
session.sql(
    """
    CREATE TABLE IF NOT EXISTS BD_AA_DEV.SC_STORAGE_BMX_PS.HYPERPARAMETER_RESULTS (
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

# Define excluded columns (metadata columns from Feature Store)
excluded_cols = [
    "customer_id",
    "brand_pres_ret",
    "week",
    "FEATURE_TIMESTAMP",  # Feature Store timestamp column
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


def create_train_func(group_name, feature_cols, X_val, y_val):
    """
    Create a training function for the Tuner.
    This function will be called for each trial with different hyperparameters.
    """
    def train_func():
        tuner_context = get_tuner_context()
        params = tuner_context.get_hyper_params()
        dm = tuner_context.get_dataset_map()
        
        # Load data from DataConnector
        train_pd = dm["train"].to_pandas()
        
        # Prepare features and target
        X_train = train_pd[feature_cols].fillna(0)
        y_train = train_pd["uni_box_week"].fillna(0)
        
        # Build model with hyperparameters from tuner
        xgb_params = params.copy()
        xgb_params["random_state"] = 42
        xgb_params["n_jobs"] = -1
        xgb_params["objective"] = "reg:squarederror"
        xgb_params["eval_metric"] = "rmse"
        
        # Train model
        model = XGBRegressor(**xgb_params)
        model.fit(X_train, y_train)
        
        # Evaluate on validation set (using outer scope variables)
        y_val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        
        # Report metrics (negative RMSE for minimization)
        tuner_context.report(metrics={"rmse": val_rmse, "mae": val_mae}, model=model)
    
    return train_func


# Process each group
# IMPORTANTE: Cada iteración procesa UN grupo específico y guarda sus hiperparámetros
# La conexión grupo-hiperparámetros se hace mediante el campo 'group_name' que se guarda
# en la tabla HYPERPARAMETER_RESULTS. Luego en el script 04, se usa este 'group_name'
# para cargar los hiperparámetros correctos para cada modelo.
for group_idx, group_name in enumerate(groups_list, 1):
    print("\n" + "=" * 80)
    print(f"🔍 Processing Group {group_idx}/{len(groups_list)}: {group_name}")
    print("=" * 80)
    
    # Filter data for this group
    # FILTRO CLAVE: Solo procesamos datos donde stats_ntile_group == group_name
    # Esto asegura que cada búsqueda de hiperparámetros usa SOLO datos de ese grupo
    group_df = train_df.filter(train_df["stats_ntile_group"] == group_name)
    group_count = group_df.count()
    
    print(f"\n📊 Group data: {group_count:,} records")
    
    if group_count < 50:
        print(f"⚠️  WARNING: Group has less than 50 records. Skipping hyperparameter search.")
        print(f"   Will use default hyperparameters for this group.")
        continue
    
    # Sample data for this group (or use full group if sample_rate = 1.0)
    if sample_rate < 1.0:
        sampled_group_df = group_df.sample(fraction=sample_rate, seed=42)
        sampled_count = sampled_group_df.count()
        print(f"   Sampled: {sampled_count:,} records ({sample_rate*100:.0f}% of {group_count:,} total)")
    else:
        # Use full group for better hyperparameter search
        sampled_group_df = group_df
        sampled_count = group_count
        print(f"   Using full group: {sampled_count:,} records (100% of group)")
    
    # Convert to pandas
    print("   Converting to pandas...")
    df_group = sampled_group_df.to_pandas()
    
    # Prepare X and y
    X = df_group[feature_cols].fillna(0)
    y = df_group["uni_box_week"].fillna(0)
    
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    print(f"   Target range: [{y.min():.2f}, {y.max():.2f}], mean: {y.mean():.2f}")
    
    # Split into train and validation sets
    if len(X) < 20:
        print(f"⚠️  WARNING: Not enough data for train/val split. Skipping.")
        continue
        
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Train: {X_train.shape[0]:,} samples, Val: {X_val.shape[0]:,} samples")
    
    # Prepare data for Tuner using DataConnector
    # Combine features and target for train set
    train_data = X_train.copy()
    train_data["uni_box_week"] = y_train.values
    
    train_dc = DataConnector.from_dataframe(train_data)
    dataset_map = {"train": train_dc}
    
    # Create training function for this group
    train_func = create_train_func(group_name, feature_cols, X_val, y_val)
    
    # Configure Tuner
    tuner_config = TunerConfig(
        metric="rmse",
        mode="min",  # Minimize RMSE
        search_alg=RandomSearch(),
        num_trials=num_trials,
        max_concurrent_trials=1,  # Run sequentially per group
    )
    
    # Create and run Tuner
    print(f"   ⏳ Starting Random Search using snowflake.ml.modeling.tune ({num_trials} trials)...")
    import time
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
        
        # Store results
        all_results[group_name] = {
            "best_params": best_params,
            "best_cv_rmse": best_rmse,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "sample_size": sampled_count,
        }
        
        # Save to database
        # IMPORTANTE: El campo 'group_name' es la CLAVE que conecta estos hiperparámetros
        # con el grupo específico. En el script 04, se usa este 'group_name' para cargar
        # los hiperparámetros correctos cuando se entrena cada modelo.
        search_id = f"xgb_snowflake_tune_{group_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        best_params_json = json.dumps(
            {
                k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                for k, v in best_params.items()
            }
        )
        
        best_mae_value = best_mae if best_mae is not None else None
        best_mae_sql = f"{best_mae_value:.6f}" if best_mae_value is not None else "NULL"
        
        # Guardar resultados con group_name como identificador del grupo
        # Este group_name será usado en script 04 para mapear hiperparámetros -> modelo
        insert_sql = f"""
            INSERT INTO BD_AA_DEV.SC_STORAGE_BMX_PS.HYPERPARAMETER_RESULTS
            (search_id, group_name, algorithm, best_params, best_cv_rmse, best_cv_mae, val_rmse, val_mae, n_iter, sample_size)
            VALUES (
                '{search_id}',
                '{group_name}',  -- ← CLAVE: Identificador del grupo (stats_ntile_group)
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
        
    except Exception as e:
        print(f"   ❌ Error during hyperparameter search: {str(e)[:200]}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()[:300]}")
        print(f"   Will use default hyperparameters for this group.")
        continue

print(f"\n✅ Completed hyperparameter search for {len(all_results)}/{len(groups_list)} groups")

# %% [markdown]
# ## 5. Summary of All Results

# %%
print("\n" + "=" * 80)
print("📊 SUMMARY OF ALL HYPERPARAMETER SEARCHES")
print("=" * 80)

# Get all saved results
summary_df = session.sql(
    """
    SELECT 
        group_name,
        best_cv_rmse,
        val_rmse,
        val_mae,
        sample_size,
        created_at
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.HYPERPARAMETER_RESULTS
    WHERE created_at >= DATEADD(HOUR, -1, CURRENT_TIMESTAMP())
    ORDER BY group_name
"""
)

print("\n📊 Results by Group:")
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
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.HYPERPARAMETER_RESULTS
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
print(f"   ✅ Search method: Snowflake ML tune.search RandomSearch (per group)")
print(f"   ✅ Groups processed: {len(all_results)}/{len(groups_list)}")
print(f"   ✅ Trials per group: {num_trials}")
print(f"   ✅ Sample rate per group: {sample_rate*100:.0f}%")

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
