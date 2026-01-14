# %% [markdown]
# # ARCA Beverage Demo: Partitioned Inference (Batch)
# 
# ## Overview
# This notebook demonstrates **batch partitioned inference** using features pre-materialized from the Feature Store.
# 
# ## Why not query Feature Store directly?
# > For **batch inference**, features are materialized from the Feature Store into an optimized table.
# > This is more efficient than querying the Feature Store for each individual record.
# >
# > The Feature Store remains the **single source of truth** for feature definitions - we used it in Notebook 02 to generate `TRAINING_DATA`.
# >
# > For **real-time inference** (e.g., a customer opens the app), we would query the Feature Store directly.
# 
# ## What We'll Do:
# 1. Load inference data with pre-materialized features
# 2. Execute **single inference call** with PARTITION BY SEGMENT
# 3. Compare predictions vs actuals
# 4. Log results for ML Observability
# 
# ## Key Message:
# **Single SQL statement → 6 models execute in parallel → Unified results**

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.ml.registry import Registry
from snowflake.snowpark import functions as F
import pandas as pd
import time

session = get_active_session()

session.sql("USE WAREHOUSE ARCA_DEMO_WH").collect()
session.sql("USE DATABASE ARCA_BEVERAGE_DEMO").collect()
session.sql("USE SCHEMA ML_DATA").collect()

registry = Registry(
    session=session,
    database_name="ARCA_BEVERAGE_DEMO",
    schema_name="MODEL_REGISTRY"
)

print("✅ Connected to Snowflake")
print(f"   Database: {session.get_current_database()}")
print(f"   Schema: {session.get_current_schema()}")

# %% [markdown]
# ## 1. Load Inference Data (Pre-materialized Features)
# 
# Using features already materialized from Feature Store into TRAINING_DATA.
# Filtering for inference period (December 2025).

# %%
print("\n📊 Loading inference data with pre-materialized features...\n")

# Get inference period data - most recent week per customer
# This ensures 1:1 mapping between input and predictions
inference_df = session.sql("""
    SELECT 
        t.CUSTOMER_ID,
        t.WEEK_START_DATE,
        t.SEGMENT,
        t.SEGMENT_DESCRIPTION,
        t.CUSTOMER_TOTAL_UNITS_4W,
        t.WEEKS_WITH_PURCHASE,
        t.VOLUME_QUARTILE,
        t.WEEK_OF_YEAR,
        t.MONTH,
        t.QUARTER,
        t.TRANSACTION_COUNT,
        t.UNIQUE_PRODUCTS_PURCHASED,
        t.AVG_UNITS_PER_TRANSACTION,
        t.WEEKLY_SALES_UNITS AS ACTUAL_WEEKLY_SALES
    FROM ARCA_BEVERAGE_DEMO.ML_DATA.TRAINING_DATA t
    WHERE t.WEEK_START_DATE >= DATE('2025-12-01')
    QUALIFY ROW_NUMBER() OVER (PARTITION BY t.CUSTOMER_ID ORDER BY t.WEEK_START_DATE DESC) = 1
    ORDER BY t.SEGMENT, t.CUSTOMER_ID
""")

print(f"✅ Inference data loaded (1 row per customer - most recent week)")
print(f"   Total records: {inference_df.count():,}")
print(f"   Unique customers: {inference_df.select('CUSTOMER_ID').distinct().count():,}")

print("\n📊 Records per Segment:")
inference_df.group_by('SEGMENT').count().sort('SEGMENT').show()

print("\n📋 Sample data with features:")
inference_df.select(
    'CUSTOMER_ID', 'SEGMENT', 'CUSTOMER_TOTAL_UNITS_4W', 
    'WEEKS_WITH_PURCHASE', 'VOLUME_QUARTILE', 'ACTUAL_WEEKLY_SALES'
).show(5)

# %% [markdown]
# ## 2. Verify Partitioned Model

# %%
print("\n📦 Verifying partitioned model...\n")

model_ref = registry.get_model("WEEKLY_SALES_FORECAST_PARTITIONED")
model_version = model_ref.version("PRODUCTION")

print("✅ Model: WEEKLY_SALES_FORECAST_PARTITIONED")
print(f"   Version: {model_version.version_name}")
print(f"   Alias: PRODUCTION")

# Show model functions
functions = session.sql("""
    SHOW FUNCTIONS IN MODEL ARCA_BEVERAGE_DEMO.MODEL_REGISTRY.WEEKLY_SALES_FORECAST_PARTITIONED
""").collect()

print(f"\n📋 Available functions:")
for f in functions:
    print(f"   - {f['name']}")

# %% [markdown]
# ## 3. Execute Partitioned Inference (SQL)
# 
# **Key syntax:** `TABLE(model!PREDICT(...) OVER (PARTITION BY SEGMENT))`
# 
# Snowflake automatically:
# 1. Splits data by SEGMENT
# 2. Routes each partition to correct sub-model
# 3. Executes in parallel
# 4. Returns unified results

# %%
print("\n" + "="*80)
print("🚀 EXECUTING PARTITIONED INFERENCE")
print("="*80)

print("\n📝 Preparing inference data...")
inference_df.write.mode('overwrite').save_as_table(
    'ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_INPUT_TEMP'
)

print("\n🔮 Running partitioned inference...\n")
print("   Syntax: TABLE(model!PREDICT(...) OVER (PARTITION BY SEGMENT))")
print("   This single call routes to 6 different sub-models automatically!\n")

start_time = time.time()

predictions_sql = """
WITH model_predictions AS (
    SELECT 
        p.CUSTOMER_ID,
        p.SEGMENT,
        p.PREDICTED_WEEKLY_SALES
    FROM ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_INPUT_TEMP i,
        TABLE(
            ARCA_BEVERAGE_DEMO.MODEL_REGISTRY.WEEKLY_SALES_FORECAST_PARTITIONED!PREDICT(
                i.CUSTOMER_ID,
                i.SEGMENT,
                i.CUSTOMER_TOTAL_UNITS_4W,
                i.WEEKS_WITH_PURCHASE,
                i.VOLUME_QUARTILE,
                i.WEEK_OF_YEAR,
                i.MONTH,
                i.QUARTER,
                i.TRANSACTION_COUNT,
                i.UNIQUE_PRODUCTS_PURCHASED,
                i.AVG_UNITS_PER_TRANSACTION::FLOAT
            ) OVER (PARTITION BY i.SEGMENT)
        ) p
)
SELECT 
    mp.CUSTOMER_ID,
    mp.SEGMENT,
    i.WEEK_START_DATE,
    i.SEGMENT_DESCRIPTION,
    ROUND(mp.PREDICTED_WEEKLY_SALES, 2) AS PREDICTED_WEEKLY_SALES,
    i.ACTUAL_WEEKLY_SALES,
    ROUND(mp.PREDICTED_WEEKLY_SALES - i.ACTUAL_WEEKLY_SALES, 2) AS PREDICTION_ERROR,
    ROUND(ABS(mp.PREDICTED_WEEKLY_SALES - i.ACTUAL_WEEKLY_SALES), 2) AS ABSOLUTE_ERROR
FROM model_predictions mp
JOIN ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_INPUT_TEMP i 
    ON mp.CUSTOMER_ID = i.CUSTOMER_ID AND mp.SEGMENT = i.SEGMENT
ORDER BY mp.SEGMENT, mp.CUSTOMER_ID
"""

predictions_df = session.sql(predictions_sql)
prediction_count = predictions_df.count()
inference_time = time.time() - start_time

print(f"✅ Inference complete!")
print(f"   ⏱️  Time: {inference_time:.2f} seconds")
print(f"   📊 Predictions: {prediction_count:,}")

print("\n📊 Sample Predictions vs Actuals:")
predictions_df.show(10)

# %% [markdown]
# ## 4. Analyze Prediction Performance by Segment

# %%
print("\n📊 Performance Analysis by Segment\n")

performance_sql = """
WITH model_predictions AS (
    SELECT 
        p.CUSTOMER_ID,
        p.SEGMENT,
        p.PREDICTED_WEEKLY_SALES
    FROM ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_INPUT_TEMP i,
        TABLE(
            ARCA_BEVERAGE_DEMO.MODEL_REGISTRY.WEEKLY_SALES_FORECAST_PARTITIONED!PREDICT(
                i.CUSTOMER_ID, i.SEGMENT,
                i.CUSTOMER_TOTAL_UNITS_4W, i.WEEKS_WITH_PURCHASE, i.VOLUME_QUARTILE,
                i.WEEK_OF_YEAR, i.MONTH, i.QUARTER,
                i.TRANSACTION_COUNT, i.UNIQUE_PRODUCTS_PURCHASED,
                i.AVG_UNITS_PER_TRANSACTION::FLOAT
            ) OVER (PARTITION BY i.SEGMENT)
        ) p
),
predictions AS (
    SELECT 
        mp.SEGMENT,
        mp.PREDICTED_WEEKLY_SALES,
        i.ACTUAL_WEEKLY_SALES,
        mp.PREDICTED_WEEKLY_SALES - i.ACTUAL_WEEKLY_SALES AS ERROR
    FROM model_predictions mp
    JOIN ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_INPUT_TEMP i 
        ON mp.CUSTOMER_ID = i.CUSTOMER_ID AND mp.SEGMENT = i.SEGMENT
)
SELECT
    SEGMENT,
    COUNT(*) AS PREDICTIONS,
    ROUND(AVG(PREDICTED_WEEKLY_SALES), 1) AS AVG_PREDICTED,
    ROUND(AVG(ACTUAL_WEEKLY_SALES), 1) AS AVG_ACTUAL,
    ROUND(AVG(ABS(ERROR)), 2) AS MAE,
    ROUND(SQRT(AVG(POWER(ERROR, 2))), 2) AS RMSE
FROM predictions
GROUP BY SEGMENT
ORDER BY SEGMENT
"""

print("By Segment:")
session.sql(performance_sql).show()

# Overall performance
overall_sql = """
WITH model_predictions AS (
    SELECT 
        p.CUSTOMER_ID,
        p.SEGMENT,
        p.PREDICTED_WEEKLY_SALES
    FROM ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_INPUT_TEMP i,
        TABLE(
            ARCA_BEVERAGE_DEMO.MODEL_REGISTRY.WEEKLY_SALES_FORECAST_PARTITIONED!PREDICT(
                i.CUSTOMER_ID, i.SEGMENT,
                i.CUSTOMER_TOTAL_UNITS_4W, i.WEEKS_WITH_PURCHASE, i.VOLUME_QUARTILE,
                i.WEEK_OF_YEAR, i.MONTH, i.QUARTER,
                i.TRANSACTION_COUNT, i.UNIQUE_PRODUCTS_PURCHASED,
                i.AVG_UNITS_PER_TRANSACTION::FLOAT
            ) OVER (PARTITION BY i.SEGMENT)
        ) p
),
predictions AS (
    SELECT 
        mp.PREDICTED_WEEKLY_SALES,
        i.ACTUAL_WEEKLY_SALES,
        mp.PREDICTED_WEEKLY_SALES - i.ACTUAL_WEEKLY_SALES AS ERROR
    FROM model_predictions mp
    JOIN ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_INPUT_TEMP i 
        ON mp.CUSTOMER_ID = i.CUSTOMER_ID AND mp.SEGMENT = i.SEGMENT
)
SELECT
    COUNT(*) AS TOTAL_PREDICTIONS,
    ROUND(AVG(ABS(ERROR)), 2) AS OVERALL_MAE,
    ROUND(SQRT(AVG(POWER(ERROR, 2))), 2) AS OVERALL_RMSE,
    ROUND(AVG(ABS(ERROR) / NULLIF(ACTUAL_WEEKLY_SALES, 0)) * 100, 1) AS MAPE_PCT
FROM predictions
"""

print("\nOverall Performance:")
session.sql(overall_sql).show()

# %% [markdown]
# ## 5. Save Predictions to Inference Logs

# %%
print("\n💾 Saving predictions to inference logs...\n")

# Create inference logs table
session.sql("""
CREATE TABLE IF NOT EXISTS ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_LOGS (
    CUSTOMER_ID NUMBER,
    SEGMENT VARCHAR,
    WEEK_START_DATE DATE,
    PREDICTED_WEEKLY_SALES FLOAT,
    ACTUAL_WEEKLY_SALES FLOAT,
    PREDICTION_ERROR FLOAT,
    ABSOLUTE_ERROR FLOAT,
    INFERENCE_TIMESTAMP TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    MODEL_VERSION VARCHAR
)
""").collect()

# Clear previous logs (optional - for demo purposes)
session.sql("DELETE FROM ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_LOGS").collect()

# Insert predictions using CTE + JOIN pattern
session.sql(f"""
INSERT INTO ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_LOGS
    (CUSTOMER_ID, SEGMENT, WEEK_START_DATE, PREDICTED_WEEKLY_SALES, 
     ACTUAL_WEEKLY_SALES, PREDICTION_ERROR, ABSOLUTE_ERROR, MODEL_VERSION)
WITH model_predictions AS (
    SELECT 
        p.CUSTOMER_ID,
        p.SEGMENT,
        p.PREDICTED_WEEKLY_SALES
    FROM ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_INPUT_TEMP i,
        TABLE(
            ARCA_BEVERAGE_DEMO.MODEL_REGISTRY.WEEKLY_SALES_FORECAST_PARTITIONED!PREDICT(
                i.CUSTOMER_ID, i.SEGMENT,
                i.CUSTOMER_TOTAL_UNITS_4W, i.WEEKS_WITH_PURCHASE, i.VOLUME_QUARTILE,
                i.WEEK_OF_YEAR, i.MONTH, i.QUARTER,
                i.TRANSACTION_COUNT, i.UNIQUE_PRODUCTS_PURCHASED,
                i.AVG_UNITS_PER_TRANSACTION::FLOAT
            ) OVER (PARTITION BY i.SEGMENT)
        ) p
)
SELECT 
    mp.CUSTOMER_ID,
    mp.SEGMENT,
    i.WEEK_START_DATE,
    mp.PREDICTED_WEEKLY_SALES,
    i.ACTUAL_WEEKLY_SALES,
    mp.PREDICTED_WEEKLY_SALES - i.ACTUAL_WEEKLY_SALES,
    ABS(mp.PREDICTED_WEEKLY_SALES - i.ACTUAL_WEEKLY_SALES),
    '{model_version.version_name}'
FROM model_predictions mp
JOIN ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_INPUT_TEMP i 
    ON mp.CUSTOMER_ID = i.CUSTOMER_ID AND mp.SEGMENT = i.SEGMENT
""").collect()

log_count = session.sql("SELECT COUNT(*) as CNT FROM ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_LOGS").collect()[0]['CNT']
print(f"✅ Saved {log_count:,} predictions to INFERENCE_LOGS")

print("\n📋 Sample from logs:")
session.sql("""
    SELECT * FROM ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_LOGS 
    ORDER BY SEGMENT, CUSTOMER_ID 
    LIMIT 5
""").show()

# %% [markdown]
# ## 6. SQL-Only Demo (Copy-Paste Ready)
# 
# This query can be executed from **any SQL client** - Snowsight, Python, JDBC, etc.

# %%
sql_demo = """
-- ============================================================
-- PARTITIONED INFERENCE - SINGLE SQL CALL
-- ============================================================
-- This query:
--   1. Takes customer features as input
--   2. Routes each segment to its specific model
--   3. Returns predictions for ALL segments in one call
-- ============================================================

WITH model_predictions AS (
    SELECT 
        p.CUSTOMER_ID,
        p.SEGMENT,
        p.PREDICTED_WEEKLY_SALES
    FROM ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_INPUT_TEMP input,
        TABLE(
            ARCA_BEVERAGE_DEMO.MODEL_REGISTRY.WEEKLY_SALES_FORECAST_PARTITIONED!PREDICT(
                input.CUSTOMER_ID,
                input.SEGMENT,
                input.CUSTOMER_TOTAL_UNITS_4W,
                input.WEEKS_WITH_PURCHASE,
                input.VOLUME_QUARTILE,
                input.WEEK_OF_YEAR,
                input.MONTH,
                input.QUARTER,
                input.TRANSACTION_COUNT,
                input.UNIQUE_PRODUCTS_PURCHASED,
                input.AVG_UNITS_PER_TRANSACTION::FLOAT
            ) OVER (PARTITION BY input.SEGMENT)  -- ← THIS IS THE KEY!
        ) p
)
SELECT 
    mp.CUSTOMER_ID,
    mp.SEGMENT,
    i.WEEK_START_DATE,
    ROUND(mp.PREDICTED_WEEKLY_SALES, 2) AS PREDICTED_WEEKLY_SALES,
    i.ACTUAL_WEEKLY_SALES
FROM model_predictions mp
JOIN ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_INPUT_TEMP i 
    ON mp.CUSTOMER_ID = i.CUSTOMER_ID AND mp.SEGMENT = i.SEGMENT
ORDER BY mp.SEGMENT, mp.CUSTOMER_ID
LIMIT 20;
"""

print("📋 SQL Query for Demo (copy-paste ready):")
print("="*60)
print(sql_demo)
print("="*60)
print("\n💡 Key points:")
print("   1. OVER (PARTITION BY SEGMENT) routes to correct sub-model")
print("   2. CTE + JOIN pattern to get original columns back")

# %% [markdown]
# ## 7. Summary

# %%
print("\n" + "="*80)
print("🎉 PARTITIONED INFERENCE COMPLETE!")
print("="*80)

print(f"""
📊 Summary:
   ✅ Predictions generated: {prediction_count:,}
   ✅ Segments covered: 6
   ✅ Inference time: {inference_time:.2f} seconds
   ✅ Logs saved to: INFERENCE_LOGS

💡 Key Advantages of Partitioned Models:
   ✅ Single model in registry (not 6 separate)
   ✅ Automatic routing by SEGMENT column
   ✅ SQL-native inference (no Python required)
   ✅ Parallel execution handled by Snowflake
   ✅ Unified results in single query

🎯 Business Impact:
   • One SQL call predicts for ALL customer segments
   • No need to manage 6 separate model endpoints
   • Easy to integrate into any BI tool or pipeline
   • Production-ready pattern for multi-segment forecasting

🚀 Next Steps:
   → Notebook 06: ML Observability (drift monitoring, alerts)
""")

print("="*80)


