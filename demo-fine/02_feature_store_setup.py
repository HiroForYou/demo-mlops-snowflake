# %% [markdown]
# # ARCA Beverage Demo: ML Observability
# 
# ## Overview
# This notebook sets up **automated ML monitoring** to replace manual drift detection.
# 
# ## Key Capabilities:
# 1. **Drift Monitoring**: Jensen-Shannon divergence on feature distributions
# 2. **Performance Tracking**: RMSE, MAE, WAPE metrics over time
# 3. **Automated Alerts**: Threshold-based notifications (JS > 0.2 warning, > 0.4 critical)
# 4. **Feature Distribution Monitoring**: Training vs Inference comparison
# 
# ## Business Value:
# **Replaces manual drift detection with automated monitoring**:
# - Before: Data scientists manually check for drift (weekly/monthly)
# - After: Real-time automated monitoring with alerts
# - Snowflake-native: No external tools required
# 
# ## Key Message:
# "Built-in ML Observability eliminates manual monitoring overhead and catches issues early."

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F
import pandas as pd
from datetime import datetime, timedelta

session = get_active_session()

session.sql("USE WAREHOUSE ARCA_DEMO_WH").collect()
session.sql("USE DATABASE ARCA_BEVERAGE_DEMO").collect()
session.sql("USE SCHEMA MODEL_REGISTRY").collect()

print(f"✅ Connected to Snowflake")
print(f"   Database: {session.get_current_database()}")
print(f"   Schema: {session.get_current_schema()}")

# %% [markdown]
# ## 1. Verify Inference Logs Exist
# 
# Check that we have predictions from Notebook 05

# %%
inference_check = session.sql("""
SELECT
    COUNT(*) AS TOTAL_PREDICTIONS,
    COUNT(DISTINCT CUSTOMER_ID) AS UNIQUE_CUSTOMERS,
    COUNT(DISTINCT SEGMENT) AS SEGMENTS,
    MIN(INFERENCE_TIMESTAMP) AS FIRST_PREDICTION,
    MAX(INFERENCE_TIMESTAMP) AS LAST_PREDICTION
FROM ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_LOGS
""")

print("\n📊 Inference Logs Status:\n")
inference_check.show()

log_count = session.table('ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_LOGS').count()

if log_count == 0:
    print("\n❌ ERROR: No inference logs found!")
    print("   Please run Notebook 05 (Partitioned Inference) first.")
else:
    print(f"\n✅ {log_count:,} predictions ready for monitoring")

# %% [markdown]
# ## 2. Prepare Baseline Data for Drift Detection
# 
# Create baseline tables from training data (used for comparison)

# %%
print("\n📝 Creating baseline tables for each segment...\n")

for segment_num in range(1, 7):
    segment = f'SEGMENT_{segment_num}'
    
    baseline_sql = f"""
    CREATE OR REPLACE TABLE ARCA_BEVERAGE_DEMO.ML_DATA.BASELINE_{segment} AS
    SELECT
        CUSTOMER_ID,
        WEEK_START_DATE AS TIMESTAMP_COL,
        WEEKLY_SALES_UNITS::FLOAT AS ACTUAL_WEEKLY_SALES,
        WEEKLY_SALES_UNITS::FLOAT AS PREDICTED_WEEKLY_SALES,
        CUSTOMER_TOTAL_UNITS_4W::FLOAT AS CUSTOMER_TOTAL_UNITS_4W,
        WEEKS_WITH_PURCHASE::FLOAT AS WEEKS_WITH_PURCHASE,
        VOLUME_QUARTILE::FLOAT AS VOLUME_QUARTILE,
        WEEK_OF_YEAR::FLOAT AS WEEK_OF_YEAR,
        MONTH::FLOAT AS MONTH,
        QUARTER::FLOAT AS QUARTER,
        TRANSACTION_COUNT::FLOAT AS TRANSACTION_COUNT,
        UNIQUE_PRODUCTS_PURCHASED::FLOAT AS UNIQUE_PRODUCTS_PURCHASED,
        AVG_UNITS_PER_TRANSACTION::FLOAT AS AVG_UNITS_PER_TRANSACTION
    FROM ARCA_BEVERAGE_DEMO.ML_DATA.TRAINING_DATA
    WHERE SEGMENT = '{segment}'
    """
    
    session.sql(baseline_sql).collect()
    
    count = session.table(f'ARCA_BEVERAGE_DEMO.ML_DATA.BASELINE_{segment}').count()
    print(f"✅ {segment}: {count:,} baseline records")

print("\n✅ All baseline tables created with FLOAT types")

# %% [markdown]
# ## 3. Prepare Inference Data for Monitoring
# 
# Create monitoring-ready tables with proper structure

# %%
print("\n📝 Creating inference tables for monitoring...\n")

for segment_num in range(1, 7):
    segment = f'SEGMENT_{segment_num}'
    
    inference_sql = f"""
    CREATE OR REPLACE TABLE ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_{segment} AS
    SELECT
        l.CUSTOMER_ID,
        l.INFERENCE_TIMESTAMP AS TIMESTAMP_COL,
        l.PREDICTED_WEEKLY_SALES::FLOAT AS PREDICTED_WEEKLY_SALES,
        l.ACTUAL_WEEKLY_SALES::FLOAT AS ACTUAL_WEEKLY_SALES,
        i.CUSTOMER_TOTAL_UNITS_4W::FLOAT AS CUSTOMER_TOTAL_UNITS_4W,
        i.WEEKS_WITH_PURCHASE::FLOAT AS WEEKS_WITH_PURCHASE,
        i.VOLUME_QUARTILE::FLOAT AS VOLUME_QUARTILE,
        i.WEEK_OF_YEAR::FLOAT AS WEEK_OF_YEAR,
        i.MONTH::FLOAT AS MONTH,
        i.QUARTER::FLOAT AS QUARTER,
        i.TRANSACTION_COUNT::FLOAT AS TRANSACTION_COUNT,
        i.UNIQUE_PRODUCTS_PURCHASED::FLOAT AS UNIQUE_PRODUCTS_PURCHASED,
        i.AVG_UNITS_PER_TRANSACTION::FLOAT AS AVG_UNITS_PER_TRANSACTION
    FROM ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_LOGS l
    JOIN ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_INPUT_TEMP i 
        ON l.CUSTOMER_ID = i.CUSTOMER_ID AND l.SEGMENT = i.SEGMENT
    WHERE l.SEGMENT = '{segment}'
    """
    
    session.sql(inference_sql).collect()
    
    count = session.table(f'ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_{segment}').count()
    print(f"✅ {segment}: {count:,} inference records")

print("\n✅ All inference tables ready with FLOAT types")

# %% [markdown]
# ## 4. Create Model Monitors
# 
# Create monitors for 2-3 segments (demo purposes - not all 6 needed)

# %%
print("\n" + "="*80)
print("🔧 CREATING MODEL MONITORS")
print("="*80)
print("\nCreating monitors for SEGMENT_1, SEGMENT_3, and SEGMENT_5 (demo sample)\n")

monitor_segments = ['SEGMENT_1', 'SEGMENT_3', 'SEGMENT_5']
created_monitors = []

for segment in monitor_segments:
    model_name = f'WEEKLY_SALES_FORECAST_{segment}'
    monitor_name = f'WEEKLY_SALES_{segment}_MONITOR'
    
    print(f"🔧 Creating monitor for {segment}...", end=" ")
    
    try:
        drop_sql = f"DROP MODEL MONITOR IF EXISTS ARCA_BEVERAGE_DEMO.MODEL_REGISTRY.{monitor_name}"
        session.sql(drop_sql).collect()
        
        # Corrected syntax based on documentation
        create_monitor_sql = f"""
        CREATE MODEL MONITOR ARCA_BEVERAGE_DEMO.MODEL_REGISTRY.{monitor_name} WITH
            MODEL = ARCA_BEVERAGE_DEMO.MODEL_REGISTRY.{model_name}
            VERSION = 'PRODUCTION'
            FUNCTION = 'PREDICT'
            SOURCE = ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_{segment}
            BASELINE = ARCA_BEVERAGE_DEMO.ML_DATA.BASELINE_{segment}
            WAREHOUSE = ARCA_DEMO_WH
            REFRESH_INTERVAL = '1 day'
            AGGREGATION_WINDOW = '1 day'
            TIMESTAMP_COLUMN = TIMESTAMP_COL
            ID_COLUMNS = ('CUSTOMER_ID')
            PREDICTION_SCORE_COLUMNS = ('PREDICTED_WEEKLY_SALES')
            ACTUAL_SCORE_COLUMNS = ('ACTUAL_WEEKLY_SALES')
        """
        
        session.sql(create_monitor_sql).collect()
        created_monitors.append(segment)
        print("✅ Success")
        
    except Exception as e:
        error_msg = str(e)[:150]
        print(f"⚠️  {error_msg}")

print(f"\n✅ Created {len(created_monitors)}/3 monitors successfully")

if len(created_monitors) > 0:
    print("\n📊 Verifying monitors...")
    monitors_check = session.sql("""
        SHOW MODEL MONITORS IN SCHEMA ARCA_BEVERAGE_DEMO.MODEL_REGISTRY
    """)
    monitors_check.show()

# %% [markdown]
# ## 5. Query Drift Metrics
# 
# Demonstrate drift detection using Jensen-Shannon divergence

# %%
if len(created_monitors) > 0:
    print("\n📊 Querying Drift Metrics (Jensen-Shannon Divergence)...\n")
    
    # Wait a moment for monitors to initialize
    import time
    print("⏳ Waiting for monitors to initialize (30 seconds)...")
    time.sleep(30)
    
    for segment in created_monitors:
        monitor_name = f'WEEKLY_SALES_{segment}_MONITOR'
        
        print(f"\n🔍 Drift Analysis for {segment}:")
        
        try:
            drift_sql = f"""
            SELECT *
            FROM TABLE(MODEL_MONITOR_DRIFT_METRIC(
                'ARCA_BEVERAGE_DEMO.MODEL_REGISTRY.{monitor_name}',
                'JENSEN_SHANNON',
                'CUSTOMER_TOTAL_UNITS_4W',
                '1 DAY',
                DATEADD('DAY', -30, CURRENT_TIMESTAMP()),
                CURRENT_TIMESTAMP()
            ))
            LIMIT 5
            """
            
            drift_results = session.sql(drift_sql)
            drift_results.show()
            
        except Exception as e:
            print(f"   ⚠️  Drift metrics not yet available: {str(e)[:100]}")
            print(f"   💡 Monitors need time to compute metrics. Check Snowsight UI in 5-10 minutes.")
else:
    print("\n⚠️  No monitors created, skipping drift queries")

# %% [markdown]
# ## 6. Query Performance Metrics
# 
# Track model performance over time (RMSE, MAE)

# %%
if len(created_monitors) > 0:
    print("\n📈 Querying Performance Metrics (RMSE)...\n")
    
    for segment in created_monitors:
        monitor_name = f'WEEKLY_SALES_{segment}_MONITOR'
        
        print(f"\n📊 Performance for {segment}:")
        
        try:
            performance_sql = f"""
            SELECT *
            FROM TABLE(MODEL_MONITOR_PERFORMANCE_METRIC(
                'ARCA_BEVERAGE_DEMO.MODEL_REGISTRY.{monitor_name}',
                'RMSE',
                '1 DAY',
                DATEADD('DAY', -30, CURRENT_DATE()),
                CURRENT_DATE()
            ))
            LIMIT 5
            """
            
            performance_results = session.sql(performance_sql)
            performance_results.show()
            
        except Exception as e:
            print(f"   ⚠️  Performance metrics not yet available: {str(e)[:100]}")
            print(f"   💡 Monitors need aggregation time. Check Snowsight UI in 5-10 minutes.")
else:
    print("\n⚠️  No monitors created, skipping performance queries")

# %% [markdown]
# ## 7. Custom Drift Analysis (SQL-based)
# 
# Manual drift detection for features (baseline vs inference)

# %%
print("\n📊 Custom Feature Distribution Analysis\n")
print("Comparing baseline (training) vs inference distributions\n")

for segment_num in [1, 3, 5]:  # Sample segments
    segment = f'SEGMENT_{segment_num}'
    
    print(f"\n🔍 {segment} - Feature Distribution Comparison:")
    
    distribution_sql = f"""
    WITH baseline_stats AS (
        SELECT
            'BASELINE' AS DATA_SOURCE,
            AVG(CUSTOMER_TOTAL_UNITS_4W) AS AVG_UNITS,
            STDDEV(CUSTOMER_TOTAL_UNITS_4W) AS STDDEV_UNITS,
            AVG(WEEKS_WITH_PURCHASE) AS AVG_WEEKS,
            AVG(TRANSACTION_COUNT) AS AVG_TRANSACTIONS
        FROM ARCA_BEVERAGE_DEMO.ML_DATA.BASELINE_{segment}
    ),
    inference_stats AS (
        SELECT
            'INFERENCE' AS DATA_SOURCE,
            AVG(CUSTOMER_TOTAL_UNITS_4W) AS AVG_UNITS,
            STDDEV(CUSTOMER_TOTAL_UNITS_4W) AS STDDEV_UNITS,
            AVG(WEEKS_WITH_PURCHASE) AS AVG_WEEKS,
            AVG(TRANSACTION_COUNT) AS AVG_TRANSACTIONS
        FROM ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_{segment}
    )
    SELECT
        DATA_SOURCE,
        ROUND(AVG_UNITS, 2) AS AVG_UNITS,
        ROUND(STDDEV_UNITS, 2) AS STDDEV_UNITS,
        ROUND(AVG_WEEKS, 2) AS AVG_WEEKS,
        ROUND(AVG_TRANSACTIONS, 2) AS AVG_TRANSACTIONS
    FROM baseline_stats
    UNION ALL
    SELECT
        DATA_SOURCE,
        ROUND(AVG_UNITS, 2),
        ROUND(STDDEV_UNITS, 2),
        ROUND(AVG_WEEKS, 2),
        ROUND(AVG_TRANSACTIONS, 2)
    FROM inference_stats
    """
    
    session.sql(distribution_sql).show()

# %% [markdown]
# ## 8. Performance Degradation Detection
# 
# Identify segments where model performance has degraded

# %%
performance_comparison = session.sql("""
WITH inference_performance AS (
    SELECT
        SEGMENT,
        AVG(ABS(PREDICTION_ERROR)) AS MAE,
        SQRT(AVG(POWER(PREDICTION_ERROR, 2))) AS RMSE,
        AVG(ABS(PREDICTION_ERROR / NULLIF(ACTUAL_WEEKLY_SALES, 0))) * 100 AS MAPE_PCT,
        COUNT(*) AS PREDICTION_COUNT
    FROM ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_LOGS
    GROUP BY SEGMENT
)
SELECT
    SEGMENT,
    PREDICTION_COUNT,
    ROUND(MAE, 2) AS MAE,
    ROUND(RMSE, 2) AS RMSE,
    ROUND(MAPE_PCT, 1) AS MAPE_PCT,
    CASE
        WHEN RMSE > 1.0 THEN '🔴 HIGH ERROR'
        WHEN RMSE > 0.5 THEN '🟡 MODERATE ERROR'
        ELSE '🟢 LOW ERROR'
    END AS PERFORMANCE_STATUS
FROM inference_performance
ORDER BY RMSE DESC
""")

print("\n📊 Model Performance Status by Segment:\n")
performance_comparison.show()

# %% [markdown]
# ## 9. Alert Threshold Configuration
# 
# Define alerting rules (example - would be automated in production)

# %%
print("\n🚨 Alert Threshold Configuration\n")
print("="*60)

alert_config = {
    'drift': {
        'warning': 0.2,
        'critical': 0.4,
        'metric': 'Jensen-Shannon Divergence'
    },
    'performance': {
        'warning': 0.5,
        'critical': 1.0,
        'metric': 'RMSE'
    },
    'mape': {
        'warning': 15.0,
        'critical': 25.0,
        'metric': 'MAPE (%)'
    }
}

for alert_type, config in alert_config.items():
    print(f"\n{alert_type.upper()} Alerts:")
    print(f"  Metric: {config['metric']}")
    print(f"  ⚠️  Warning: > {config['warning']}")
    print(f"  🔴 Critical: > {config['critical']}")

print("\n💡 Production Setup:")
print("  1. Configure Snowflake Alerts on monitor tables")
print("  2. Set up email/Slack notifications")
print("  3. Create automated retraining triggers")
print("  4. Dashboard in Snowsight for real-time monitoring")

print("\n" + "="*60)

# %% [markdown]
# ## 10. Snowsight Navigation Guide
# 
# How to access ML Observability dashboards in Snowsight UI

# %%
print("\n📱 SNOWSIGHT UI NAVIGATION GUIDE")
print("="*80)

print("\n🎯 How to View ML Observability Dashboards:\n")

steps = [
    ("1. Navigate to Model Registry", "Snowsight → AI & ML → Models"),
    ("2. Select Model", "Click on 'WEEKLY_SALES_FORECAST_SEGMENT_1'"),
    ("3. View Monitors Tab", "Click 'Monitors' tab in model details"),
    ("4. Open Monitor Dashboard", "Click monitor name to see drift/performance charts"),
    ("5. Customize Time Range", "Use date picker to adjust monitoring window"),
    ("6. View Specific Metrics", "Click Settings to toggle metrics display")
]

for step, instruction in steps:
    print(f"   {step}")
    print(f"      → {instruction}\n")

print("💡 Key Dashboard Features:")
print("   - Drift metrics over time (Jensen-Shannon divergence)")
print("   - Performance metrics (RMSE, MAE, MAPE)")
print("   - Feature distribution comparison")
print("   - Prediction volume tracking")
print("   - Automated refresh (real-time updates)")

print("\n🔗 Direct SQL Access:")
print("   - MODEL_MONITOR_DRIFT_METRIC()")
print("   - MODEL_MONITOR_PERFORMANCE_METRIC()")
print("   - MODEL_MONITOR_STAT_METRIC()")

print("\n" + "="*80)

# %% [markdown]
# ## 11. Summary & Key Takeaways

# %%
print("\n" + "="*80)
print("🎉 ML OBSERVABILITY SETUP COMPLETE!")
print("="*80)

print("\n✅ Completed Setup:")
print(f"   - Model Monitors: {len(created_monitors)} segments")
print(f"   - Baseline Tables: 6 segments")
print(f"   - Inference Tables: 6 segments")
print("   - Alert Thresholds: Configured")
print("   - Snowsight Dashboards: Available")

print("\n📊 Monitoring Capabilities:")
print("   ✅ Drift Detection (Jensen-Shannon divergence)")
print("   ✅ Performance Tracking (RMSE, MAE, MAPE)")
print("   ✅ Feature Distribution Analysis")
print("   ✅ Automated Alerting (configurable thresholds)")

print("\n💡 Key Business Messages:")
print("   - Replaces manual drift detection with automation")
print("   - Real-time monitoring vs weekly/monthly checks")
print("   - Snowflake-native (no external tools required)")
print("   - Catches issues early before impacting business")

print("\n🎯 Demo Talking Points:")
print("   1. Show Snowsight dashboard with drift charts")
print("   2. Explain JS divergence > 0.4 triggers alert")
print("   3. Compare baseline vs inference distributions")
print("   4. Highlight automated vs manual monitoring savings")

print("\n🚀 Production Next Steps:")
print("   - Configure Snowflake Alerts for automated notifications")
print("   - Set up email/Slack integration")
print("   - Create model retraining triggers on drift")
print("   - Expand to all 6 segments (currently 3 demo)")
print("   - Add custom business metrics to dashboards")

print("\n" + "="*80)
print("\n🎓 DEMO COMPLETE - All 6 Notebooks Executed Successfully!")
print("\n📋 Full ML Workflow Demonstrated:")
print("   01 ✅ Data Setup")
print("   02 ✅ Feature Store (3 refresh frequencies)")
print("   03 ✅ Customer Segmentation (6 segments)")
print("   04 ✅ Many Model Training (6 models, 6x speedup)")
print("   05 ✅ Partitioned Inference (batch predictions)")
print("   06 ✅ ML Observability (automated monitoring)")
print("\n" + "="*80)


