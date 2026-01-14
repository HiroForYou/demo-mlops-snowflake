# %% [markdown]
# # Migration: Hyperparameter Search (XGBoost + Random Search)
# 
# ## Overview
# This script performs hyperparameter optimization using Random Search for XGBoost regression.
# 
# ## What We'll Do:
# 1. Load cleaned training data (sampled for efficiency)
# 2. Prepare features and target
# 3. Perform Random Search with cross-validation
# 4. Save best hyperparameters

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import randint, uniform
import pickle
from datetime import datetime

session = get_active_session()

# Set context
session.sql("USE WAREHOUSE ARCA_DEMO_WH").collect()
session.sql("USE DATABASE BD_AA_DEV").collect()
session.sql("USE SCHEMA SC_STORAGE_BMX_PS").collect()

print(f"✅ Connected to Snowflake")
print(f"   Database: {session.get_current_database()}")
print(f"   Schema: {session.get_current_schema()}")

# %% [markdown]
# ## 1. Load and Sample Training Data

# %%
print("\n" + "="*80)
print("📊 LOADING TRAINING DATA")
print("="*80)

# Load cleaned training data
train_df = session.table("BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED")

total_rows = train_df.count()
print(f"\n✅ Training data loaded")
print(f"   Total rows: {total_rows:,}")

# Sample data for hyperparameter search (5-10% for efficiency)
sample_rate = 0.05  # 5% sample
print(f"\n📊 Sampling {sample_rate*100:.0f}% of data for hyperparameter search...")

sampled_df = train_df.sample(fraction=sample_rate, seed=42)
sampled_count = sampled_df.count()
print(f"   Sampled rows: {sampled_count:,}")

# Convert to pandas for sklearn
print("\n⏳ Converting to pandas (this may take a moment)...")
df = sampled_df.to_pandas()
print(f"✅ Converted to pandas: {df.shape}")

# %% [markdown]
# ## 2. Prepare Features and Target

# %%
print("\n" + "="*80)
print("🔧 PREPARING FEATURES AND TARGET")
print("="*80)

# Define excluded columns
excluded_cols = [
    'customer_id', 'brand_pres_ret', 'week', 
    'group', 'stats_group', 'percentile_group', 'stats_ntile_group'
]

# Get feature columns (all except excluded and target)
feature_cols = [col for col in df.columns 
                if col not in excluded_cols + ['uni_box_week']]

print(f"\n📋 Features ({len(feature_cols)}):")
for col in sorted(feature_cols):
    print(f"   - {col}")

# Prepare X and y
X = df[feature_cols].fillna(0)  # Fill NaN with 0
y = df['uni_box_week'].fillna(0)

print(f"\n✅ Features prepared:")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print(f"   Target range: [{y.min():.2f}, {y.max():.2f}]")
print(f"   Target mean: {y.mean():.2f}")

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Train/Validation split:")
print(f"   Training: {X_train.shape[0]:,} samples")
print(f"   Validation: {X_val.shape[0]:,} samples")

# %% [markdown]
# ## 3. Define Hyperparameter Search Space

# %%
print("\n" + "="*80)
print("🎯 DEFINING HYPERPARAMETER SEARCH SPACE")
print("="*80)

# Define parameter distributions for Random Search
param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 1.0),
    'colsample_bytree': uniform(0.6, 1.0),
    'min_child_weight': randint(1, 7),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

print("\n📋 Hyperparameter Search Space:")
for param, dist in param_distributions.items():
    if hasattr(dist, 'a') and hasattr(dist, 'b'):
        print(f"   {param}: uniform({dist.a:.2f}, {dist.b:.2f})")
    elif hasattr(dist, 'low') and hasattr(dist, 'high'):
        print(f"   {param}: randint({dist.low}, {dist.high})")

# Number of iterations for Random Search
n_iter = 50
print(f"\n🔢 Random Search iterations: {n_iter}")

# %% [markdown]
# ## 4. Perform Random Search

# %%
print("\n" + "="*80)
print("🔍 PERFORMING RANDOM SEARCH")
print("="*80)

# Create base XGBoost model
base_model = XGBRegressor(
    random_state=42,
    n_jobs=-1,
    objective='reg:squarederror',
    eval_metric='rmse'
)

# Create Random Search
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=n_iter,
    cv=5,  # 5-fold cross-validation
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("\n⏳ Starting Random Search (this may take several minutes)...")
print("   Using 5-fold cross-validation")
print("   This will test 50 different hyperparameter combinations\n")

# Fit Random Search
import time
start_time = time.time()

random_search.fit(X_train, y_train)

elapsed_time = time.time() - start_time
print(f"\n✅ Random Search completed in {elapsed_time/60:.2f} minutes")

# %% [markdown]
# ## 5. Evaluate Best Model

# %%
print("\n" + "="*80)
print("🏆 BEST HYPERPARAMETERS")
print("="*80)

best_params = random_search.best_params_
best_score = random_search.best_score_
best_model = random_search.best_estimator_

print("\n📊 Best Hyperparameters:")
for param, value in sorted(best_params.items()):
    print(f"   {param}: {value}")

print(f"\n📈 Best CV Score (neg MSE): {best_score:.4f}")
print(f"   Best CV RMSE: {np.sqrt(-best_score):.4f}")

# Evaluate on validation set
y_val_pred = best_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)

print(f"\n📊 Validation Set Performance:")
print(f"   RMSE: {val_rmse:.4f}")
print(f"   MAE: {val_mae:.4f}")

# %% [markdown]
# ## 6. Save Hyperparameters

# %%
print("\n" + "="*80)
print("💾 SAVING HYPERPARAMETERS")
print("="*80)

# Create results table
session.sql("""
    CREATE TABLE IF NOT EXISTS BD_AA_DEV.SC_STORAGE_BMX_PS.HYPERPARAMETER_RESULTS (
        search_id VARCHAR,
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
""").collect()

# Save results
search_id = f"xgb_random_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Convert best_params to JSON string for VARIANT type
import json
best_params_json = json.dumps({k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                               for k, v in best_params.items()})

insert_sql = f"""
    INSERT INTO BD_AA_DEV.SC_STORAGE_BMX_PS.HYPERPARAMETER_RESULTS
    (search_id, algorithm, best_params, best_cv_rmse, best_cv_mae, val_rmse, val_mae, n_iter, sample_size)
    VALUES (
        '{search_id}',
        'XGBoost',
        PARSE_JSON('{best_params_json}'),
        {np.sqrt(-best_score):.6f},
        NULL,
        {val_rmse:.6f},
        {val_mae:.6f},
        {n_iter},
        {sampled_count}
    )
"""

session.sql(insert_sql).collect()

print(f"✅ Hyperparameters saved to HYPERPARAMETER_RESULTS")
print(f"   Search ID: {search_id}")

# Verify save
saved_results = session.sql(f"""
    SELECT 
        search_id,
        algorithm,
        best_cv_rmse,
        val_rmse,
        val_mae,
        created_at
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.HYPERPARAMETER_RESULTS
    WHERE search_id = '{search_id}'
""")

print("\n📊 Saved Results:")
saved_results.show()

# %% [markdown]
# ## 7. Summary

# %%
print("\n" + "="*80)
print("✅ HYPERPARAMETER SEARCH COMPLETE!")
print("="*80)

print("\n📋 Summary:")
print(f"   ✅ Algorithm: XGBoost")
print(f"   ✅ Search method: Random Search")
print(f"   ✅ Iterations: {n_iter}")
print(f"   ✅ Sample size: {sampled_count:,} ({sample_rate*100:.0f}% of total)")
print(f"   ✅ Best CV RMSE: {np.sqrt(-best_score):.4f}")
print(f"   ✅ Validation RMSE: {val_rmse:.4f}")
print(f"   ✅ Search ID: {search_id}")

print("\n💡 Next Steps:")
print("   1. Review best hyperparameters")
print("   2. Run 04_many_model_training.py to train model with best hyperparameters")
print("   3. Use full dataset for final training")

print("\n" + "="*80)
