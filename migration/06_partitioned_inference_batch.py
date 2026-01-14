# %% [markdown]
# # Migration: Partitioned Inference Batch
# 
# ## Overview
# This script executes batch inference using the partitioned model with partitioned inference syntax.
# 
# ## What We'll Do:
# 1. Load inference data from cleaned table
# 2. Prepare features for inference
# 3. Execute partitioned inference using TABLE(...) OVER (PARTITION BY ...) syntax
# 4. Save predictions to inference logs
# 5. Generate statistics

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.ml.registry import Registry
from snowflake.snowpark import functions as F
import pandas as pd
import time

session = get_active_session()

# Set context
session.sql("USE DATABASE BD_AA_DEV").collect()
session.sql("USE SCHEMA SC_STORAGE_BMX_PS").collect()

registry = Registry(
    session=session,
    database_name="BD_AA_DEV",
    schema_name="MODEL_REGISTRY"
)

print("✅ Connected to Snowflake")
print(f"   Database: {session.get_current_database()}")
print(f"   Schema: {session.get_current_schema()}")

# %% [markdown]
# ## 1. Verify Partitioned Model

# %%
print("\n" + "="*80)
print("🔍 VERIFYING PARTITIONED MODEL")
print("="*80)

model_ref = registry.get_model("UNI_BOX_REGRESSION_PARTITIONED")
model_version = model_ref.version("PRODUCTION")

print("✅ Model: UNI_BOX_REGRESSION_PARTITIONED")
print(f"   Version: {model_version.version_name}")
print(f"   Alias: PRODUCTION")

# Show model functions
functions = session.sql("""
    SHOW FUNCTIONS IN MODEL BD_AA_DEV.MODEL_REGISTRY.UNI_BOX_REGRESSION_PARTITIONED
""").collect()

print(f"\n📋 Available functions:")
for f in functions:
    print(f"   - {f['name']}")

# %% [markdown]
# ## 2. Load Inference Data

# %%
print("\n" + "="*80)
print("📊 LOADING INFERENCE DATA")
print("="*80)

# Load cleaned inference data
inference_df = session.table("BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_DATASET_CLEANED")

print(f"\n✅ Inference data loaded")
print(f"   Total records: {inference_df.count():,}")
print(f"   Unique customers: {inference_df.select('customer_id').distinct().count():,}")

# Show sample
print("\n📋 Sample inference data:")
inference_df.select(
    'customer_id', 'week', 'brand_pres_ret', 
    'sum_past_12_weeks', 'week_of_year'
).show(5)

# %% [markdown]
# ## 3. Prepare Inference Input

# %%
print("\n" + "="*80)
print("🔧 PREPARING INFERENCE INPUT")
print("="*80)

# Define excluded columns
excluded_cols = [
    'customer_id', 'brand_pres_ret', 'week', 
    'group', 'stats_group', 'percentile_group', 'stats_ntile_group'
]

# Get feature columns (same as training)
inference_columns = inference_df.columns
feature_cols = [col for col in inference_columns 
                if col not in excluded_cols]

print(f"\n📋 Features for inference ({len(feature_cols)}):")
for col in sorted(feature_cols):
    print(f"   - {col}")

# Create a dummy partition column for partitioned inference
# Since we have a single model, we can use a constant partition
inference_input = inference_df.with_column("dummy_partition", F.lit("ALL"))

# Save to temporary table for inference
inference_input.write.mode('overwrite').save_as_table(
    'BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_INPUT_TEMP'
)

print(f"\n✅ Inference input prepared and saved to temporary table")
print(f"   Records: {inference_input.count():,}")

# %% [markdown]
# ## 4. Execute Partitioned Inference

# %%
print("\n" + "="*80)
print("🚀 EXECUTING PARTITIONED INFERENCE")
print("="*80)

print("\n📝 Running partitioned inference...")
print("   Syntax: TABLE(model!PREDICT(...) OVER (PARTITION BY dummy_partition))")
print("   This enables partitioned inference even with a single model\n")

start_time = time.time()

# Build feature list for PREDICT function
# We need to pass features in the same order as training
feature_list = ", ".join([f"i.{col}" for col in feature_cols])

predictions_sql = f"""
WITH model_predictions AS (
    SELECT 
        p.customer_id,
        p.predicted_uni_box_week
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_INPUT_TEMP i,
        TABLE(
            BD_AA_DEV.MODEL_REGISTRY.UNI_BOX_REGRESSION_PARTITIONED!PREDICT(
                i.customer_id,
                {feature_list}
            ) OVER (PARTITION BY i.dummy_partition)
        ) p
)
SELECT 
    mp.customer_id,
    i.week,
    i.brand_pres_ret,
    ROUND(mp.predicted_uni_box_week, 2) AS predicted_uni_box_week
FROM model_predictions mp
JOIN BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_INPUT_TEMP i 
    ON mp.customer_id = i.customer_id
ORDER BY mp.customer_id
"""

predictions_df = session.sql(predictions_sql)
prediction_count = predictions_df.count()
inference_time = time.time() - start_time

print(f"✅ Inference complete!")
print(f"   ⏱️  Time: {inference_time:.2f} seconds")
print(f"   📊 Predictions: {prediction_count:,}")

print("\n📊 Sample Predictions:")
predictions_df.show(10)

# %% [markdown]
# ## 5. Analyze Prediction Statistics

# %%
print("\n" + "="*80)
print("📈 PREDICTION STATISTICS")
print("="*80)

stats_sql = """
WITH model_predictions AS (
    SELECT 
        p.customer_id,
        p.predicted_uni_box_week
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_INPUT_TEMP i,
        TABLE(
            BD_AA_DEV.MODEL_REGISTRY.UNI_BOX_REGRESSION_PARTITIONED!PREDICT(
                i.customer_id,
                i.sum_past_12_weeks,
                i.avg_past_12_weeks,
                i.max_past_24_weeks,
                i.sum_past_24_weeks,
                i.week_of_year,
                i.avg_avg_daily_all_hours,
                i.sum_p4w,
                i.avg_past_24_weeks,
                i.pharm_super_conv,
                i.wines_liquor,
                i.groceries,
                i.max_prev2,
                i.avg_prev2,
                i.max_prev3,
                i.avg_prev3,
                i.w_m1_total,
                i.w_m2_total,
                i.w_m3_total,
                i.w_m4_total,
                i.spec_foods,
                i.prod_key,
                i.num_coolers,
                i.num_doors,
                i.max_past_4_weeks,
                i.sum_past_4_weeks,
                i.avg_past_4_weeks,
                i.max_past_12_weeks
            ) OVER (PARTITION BY i.dummy_partition)
        ) p
)
SELECT
    COUNT(*) AS TOTAL_PREDICTIONS,
    COUNT(DISTINCT customer_id) AS UNIQUE_CUSTOMERS,
    ROUND(MIN(predicted_uni_box_week), 2) AS MIN_PREDICTION,
    ROUND(MAX(predicted_uni_box_week), 2) AS MAX_PREDICTION,
    ROUND(AVG(predicted_uni_box_week), 2) AS AVG_PREDICTION,
    ROUND(STDDEV(predicted_uni_box_week), 2) AS STDDEV_PREDICTION,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY predicted_uni_box_week), 2) AS Q1,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY predicted_uni_box_week), 2) AS MEDIAN,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY predicted_uni_box_week), 2) AS Q3
FROM model_predictions
"""

print("\n📊 Overall Statistics:")
session.sql(stats_sql).show()

# %% [markdown]
# ## 6. Save Predictions to Inference Logs

# %%
print("\n" + "="*80)
print("💾 SAVING PREDICTIONS TO INFERENCE LOGS")
print("="*80)

# Create inference logs table
session.sql("""
    CREATE TABLE IF NOT EXISTS BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_LOGS (
        customer_id VARCHAR,
        week VARCHAR,
        brand_pres_ret VARCHAR,
        predicted_uni_box_week FLOAT,
        inference_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
        model_version VARCHAR
    )
""").collect()

# Clear previous logs (optional - for demo purposes)
# session.sql("DELETE FROM BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_LOGS").collect()

# Insert predictions
insert_sql = f"""
INSERT INTO BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_LOGS
    (customer_id, week, brand_pres_ret, predicted_uni_box_week, model_version)
WITH model_predictions AS (
    SELECT 
        p.customer_id,
        p.predicted_uni_box_week
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_INPUT_TEMP i,
        TABLE(
            BD_AA_DEV.MODEL_REGISTRY.UNI_BOX_REGRESSION_PARTITIONED!PREDICT(
                i.customer_id,
                {feature_list}
            ) OVER (PARTITION BY i.dummy_partition)
        ) p
)
SELECT 
    mp.customer_id,
    i.week,
    i.brand_pres_ret,
    mp.predicted_uni_box_week,
    '{model_version.version_name}'
FROM model_predictions mp
JOIN BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_INPUT_TEMP i 
    ON mp.customer_id = i.customer_id
"""

session.sql(insert_sql).collect()

log_count = session.sql("SELECT COUNT(*) as CNT FROM BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_LOGS").collect()[0]['CNT']
print(f"✅ Saved {log_count:,} predictions to INFERENCE_LOGS")

print("\n📋 Sample from logs:")
session.sql("""
    SELECT * FROM BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_LOGS 
    ORDER BY inference_timestamp DESC
    LIMIT 5
""").show()

# %% [markdown]
# ## 7. Summary

# %%
print("\n" + "="*80)
print("🎉 PARTITIONED INFERENCE BATCH COMPLETE!")
print("="*80)

print(f"""
📊 Summary:
   ✅ Predictions generated: {prediction_count:,}
   ✅ Inference time: {inference_time:.2f} seconds
   ✅ Logs saved to: INFERENCE_LOGS
   ✅ Model version: {model_version.version_name}

💡 Key Advantages of Partitioned Model:
   ✅ Single model with partitioned API
   ✅ Consistent inference syntax
   ✅ Ready for future multi-model scenarios
   ✅ SQL-native inference (no Python required)
   ✅ Parallel execution handled by Snowflake

🎯 Business Impact:
   • Batch predictions for all inference records
   • Predictions stored for monitoring and analysis
   • Ready for production deployment
   • Scalable to multiple models if needed

🚀 Next Steps:
   → Review predictions in INFERENCE_LOGS table
   → Set up monitoring and observability
   → Schedule regular batch inference runs
""")

print("="*80)
