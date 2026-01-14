# Databricks notebook source
# MAGIC %md
# MAGIC # Sales Forecast Performance Monitoring System
# MAGIC
# MAGIC Automated pipeline to evaluate sales forecast error across multiple dimensions. Combines historical sales data with ML predictions to compute business-critical KPIs.
# MAGIC
# MAGIC **Key Metrics:**
# MAGIC - **WAPE**: Weighted Mean Absolute Percentage Error
# MAGIC - **F1 Score**: Classification performance indicator
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Execution Parameters
# MAGIC | Parameter        | Description                                                                                  | Example    |
# MAGIC |------------------|----------------------------------------------------------------------------------------------|------------|
# MAGIC | `main_dir`       | Base directory for Delta Lake artifacts                                                      | `/mnt/DL/PROCESSED/MODELS_UC/` |
# MAGIC | `execution_date` | Reference date for prediction in YYYYMMDD format; **must be a Sunday** to align with weekly runs | `20251124` |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Pipeline Stages
# MAGIC 1. **Initialization**: Load parameters and calendar data
# MAGIC 2. **Data Preparation**: Build and cache sales and forecast datasets
# MAGIC 3. **Forecast Upsert**: Merge updated forecasts into Delta
# MAGIC 4. **Metric Computation**: Calculate WAPE and F1 at multiple aggregation levels
# MAGIC 5. **Metrics Upsert**: Insert new metrics into Unity Catalog

# COMMAND ----------

# DBTITLE 1,Imports
import os
from pyspark.sql import functions as F
from delta.tables import DeltaTable

# COMMAND ----------

# DBTITLE 1,Helper functions
# MAGIC %run ../../../../../utils/helper_functions

# COMMAND ----------

# DBTITLE 1,Calendar utils
# MAGIC %run ../../../../../utils/calendar

# COMMAND ----------

# DBTITLE 1,Spark utils
# MAGIC %run ../../../../../utils/spark

# COMMAND ----------

# DBTITLE 1,Delta utils
# MAGIC %run ../../../../../utils/delta

# COMMAND ----------

# DBTITLE 1,Regression metrics
# MAGIC %run ../../../../../advanced_analytics/advanced_analytics/metrics/regression_expr

# COMMAND ----------

# DBTITLE 1,Classification metrics
# MAGIC %run ../../../../../advanced_analytics/advanced_analytics/metrics/classification_expr

# COMMAND ----------

# DBTITLE 1,Execution parameters
MAIN_DIR = dbutils.widgets.get("main_dir")
EXECUTION_DATE = dbutils.widgets.get("execution_date")
ENVIRONMENT = "prd" if dbutils.secrets.get("KV", "ENV") == "PRD" else "dev"
INPUT_ENV = (
    "dev" if ENVIRONMENT == "dev" and MAIN_DIR.startswith("/mnt/DL/") else "prd"
)  # read predictions from dev or prd

# COMMAND ----------

# DBTITLE 1,Input paths
CALENDAR_PATH = "prd.gold.v_rt_calendar"
SALES_PATH = "prd.gold.v_sales_mx"
PRODUCTS_PATH = "prd.gold.v_products_mx"
CUSTOMERS_PATH = "prd.gold.v_customers_mx"
MODEL_NAME = "forecast_bpr_customer_week"
INFERENCE_DATASET_PATH = os.path.join(MAIN_DIR, MODEL_NAME.upper(), "FEATURES")
FORECAST_PATH = f"models_{INPUT_ENV}.mx.da_predictions"

# COMMAND ----------

# DBTITLE 1,Output paths
METRICS_PATH = f"models_{ENVIRONMENT}.mx.da_performance_drift"

# COMMAND ----------

# DBTITLE 1,Domain column names
RECORD_ID_COL = "record_id"
BUSINESS_UNIT_COL = "bkcc"
BUSINESS_UNIT_NAME = "MXBEB"
MODEL_URI_COL = "model_uri"
CALMONTH_COL = "calmonth"
CUSTOMER_ID_COL = "customer_id"
BPR_COL = "brand_pres_ret"
WEEK_COL = "week"
PREDICTION_COL = "prediction"
GROUND_TRUTH_COL = "uni_box"
ENTITY_MAP_COL = "entity_map"
TIMESTAMP_COL = "ldts"
AGGREGATED_COL = "aggregated_col"
AGGREGATED_VALUE_COL = "aggregated_value"
METRIC_NAME_COL = "metric_name"
METRIC_VALUE_COL = "metric_value"

# COMMAND ----------

# DBTITLE 1,Resolve calendar-derived values
calendar_df = spark.table(CALENDAR_PATH).select("date_d", "iso_year_week_d")
inference_date = compute_x_days_ago_value(EXECUTION_DATE, days_ago=7, return_col="date_key_d")
inference_week = compute_x_days_ago_value(EXECUTION_DATE, days_ago=7, return_col="iso_year_week_d")
current_month = resolve_calendar_value(inference_date, CALMONTH_COL)

# Get all dates in target week
week_calendar_df = (
    calendar_df.filter(F.col("iso_year_week_d") == inference_week)
    .select("date_d")
    .distinct()
)

# COMMAND ----------

# DBTITLE 1,Prepare dimension tables
products_df = F.broadcast(
    get_clean_products(PRODUCTS_PATH, True)
    .select(
        F.col("material_master_id").cast("long").alias("material_master_id"),
        F.col("brand_pres_ret_d").alias(BPR_COL),
    )
    .distinct()
    .na.drop()
)

customers_df = F.broadcast(
    spark.table(CUSTOMERS_PATH)
    .filter(F.col("rgm_channel_id") == "TRADICIONAL")
    .select(CUSTOMER_ID_COL, F.col("industry_size_d").alias("size"))
    .distinct()
)

inference_df = (
    spark.read.load(INFERENCE_DATASET_PATH)
    .filter(F.col(WEEK_COL) == inference_week)
    .select(F.col(CUSTOMER_ID_COL).cast("long"), BPR_COL, "stats_ntile_group")
)

# COMMAND ----------

# DBTITLE 1,Aggregate sales
sales_df = (
    spark.table(SALES_PATH)
    .join(week_calendar_df, F.col("calday") == F.col("date_d"), how="inner")
    .withColumn(
        "material_master_id",
        F.col("material_master_id").cast("long").alias("material_master_id"),
    )
    .join(products_df, "material_master_id", how="inner")
    .withColumn(CUSTOMER_ID_COL, F.col(CUSTOMER_ID_COL).cast("long"))
    .groupBy(CUSTOMER_ID_COL, BPR_COL)
    .agg(
        F.sum("uni_boxes_sold_m").alias(GROUND_TRUTH_COL),
    )
    .filter(F.col(GROUND_TRUTH_COL) > 0)
)

# COMMAND ----------

# DBTITLE 1,Merge sales and forecasts
# Load forecast predictions
forecast_df = (
    spark.table(FORECAST_PATH)
    .select(
        F.col(MODEL_URI_COL),
        F.col(ENTITY_MAP_COL)
        .getItem(CUSTOMER_ID_COL)
        .cast("long")
        .alias(CUSTOMER_ID_COL),
        F.col(ENTITY_MAP_COL).getItem(BPR_COL).alias(BPR_COL),
        F.col(ENTITY_MAP_COL).getItem(WEEK_COL).alias(WEEK_COL),
        F.col(PREDICTION_COL),
    )
    .filter(
        (F.col(WEEK_COL) == inference_week) & F.col(MODEL_URI_COL).contains(MODEL_NAME)
    )
)

# Merge sales and forecasts
forecast_and_sales_df = (
    forecast_df.join(sales_df, on=[CUSTOMER_ID_COL, BPR_COL], how="outer")
    .withColumn(GROUND_TRUTH_COL, F.coalesce(F.col(GROUND_TRUTH_COL), F.lit(0)))
    .withColumn(PREDICTION_COL, F.coalesce(F.col(PREDICTION_COL), F.lit(0)))
)

# COMMAND ----------

# DBTITLE 1,Replicate sales without model
# Distinct list of model_uris
model_list_df = (
    forecast_and_sales_df.filter(F.col(MODEL_URI_COL).isNotNull())
    .select(MODEL_URI_COL)
    .distinct()
)
model_list_b = F.broadcast(model_list_df)


# Sales that have no model
sales_without_model = forecast_and_sales_df.filter(F.col(MODEL_URI_COL).isNull()).drop(
    MODEL_URI_COL
)

synthetic = sales_without_model.crossJoin(model_list_b).withColumn(
    WEEK_COL, F.coalesce(F.col(WEEK_COL), F.lit(inference_week))
)

# Union synthetic rows back into the main frame
forecast_and_sales_df = forecast_and_sales_df.unionByName(
    synthetic, allowMissingColumns=True
)

# COMMAND ----------

# DBTITLE 1,Enrich merged data
forecast_sales_enriched_df = (
    forecast_and_sales_df.join(customers_df, on=CUSTOMER_ID_COL, how="inner")
    .join(inference_df, on=[CUSTOMER_ID_COL, BPR_COL], how="left")
    # replace nulls
    .withColumn(
        "stats_ntile_group",
        F.coalesce(F.col("stats_ntile_group"), F.lit("NOT ASSIGNED")),
    )
)

# COMMAND ----------

# DBTITLE 1,Compute metrics
# Define SQL expressions for each metric
METRIC_EXPRESSIONS = {
    "WAPE": wape_sql_expr(GROUND_TRUTH_COL, PREDICTION_COL),
    "F1": f1_sql_expr(GROUND_TRUTH_COL, PREDICTION_COL),
}

# Minimum required keys that define the metric record identity
aggregation_keys = [WEEK_COL, MODEL_URI_COL]

# Optional segmentation dimensions used to compute metrics per segment
breakdown_keys = ["size", "stats_ntile_group"]

# Compute metrics
metric_results = compute_metrics(
    df=forecast_sales_enriched_df,
    metric_sql_map=METRIC_EXPRESSIONS,
    base_group_cols=aggregation_keys,
    extra_group_cols=breakdown_keys,
    global_label="general",
)
# metric_results schema: [week, model_uri, aggregated_col, aggregated_value, metric_name, metric_value]

# COMMAND ----------

# DBTITLE 1,Create entity map
# Build entity mapping
entity_mapping = {WEEK_COL: F.lit(inference_week)}

# Compose deterministic record ID expression
metric_results = compose_record_id(
    metric_results,
    [WEEK_COL, AGGREGATED_COL, AGGREGATED_VALUE_COL, METRIC_NAME_COL],
    MODEL_URI_COL,
    RECORD_ID_COL,
)

# # Create entity map
metric_results = create_entity_map_column(
    metric_results, entity_mapping, ENTITY_MAP_COL
)

# COMMAND ----------

# DBTITLE 1,Final selection
final_metrics_df = metric_results.select(
    F.col(RECORD_ID_COL),
    F.col(MODEL_URI_COL),
    F.col(AGGREGATED_COL),
    F.col(AGGREGATED_VALUE_COL),
    F.col(METRIC_NAME_COL),
    F.col(METRIC_VALUE_COL),
    F.col(ENTITY_MAP_COL),
    F.lit(BUSINESS_UNIT_NAME).alias(BUSINESS_UNIT_COL),
    F.lit(current_month).alias(CALMONTH_COL),
    F.current_timestamp().alias(TIMESTAMP_COL),
)

# COMMAND ----------

# DBTITLE 1,Metrics upsert
delta_upsert(
    source_df=final_metrics_df,
    target_identifier=METRICS_PATH,
    merge_keys=[RECORD_ID_COL],
    partition_col_or_cols=[MODEL_URI_COL, CALMONTH_COL],
    as_metastore=True,
)
