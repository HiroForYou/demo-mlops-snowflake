# Databricks notebook source
# MAGIC %md # Forecast Predict
# MAGIC **Objective:** Load features for the target week, fetch the MLflow model by URI and write sales forecasts.
# MAGIC
# MAGIC **Inputs (Databricks widgets)**
# MAGIC - `main_dir` (str): base path where FEATURES are stored (example: `/mnt/DL/PROCESSED/MODELS_UC/FORECAST_BPR_CUSTOMER_WEEK/`).
# MAGIC - `execution_date` (str): execution date (Sunday) in `YYYYMMDD` format.
# MAGIC - `model_uri` (str): MLflow model URI, e.g. `models:/models_prd.mx.forecast_bpr_customer_week@20250910`.
# MAGIC
# MAGIC **Output**
# MAGIC - Predictions persisted into metastore table `models_prd.mx.da_predictions`.

# COMMAND ----------

# DBTITLE 1,Imports
import os
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, functions as F

# COMMAND ----------

# DBTITLE 1,Calendar utils
# MAGIC %run ../../../../utils/calendar

# COMMAND ----------

# DBTITLE 1,Logging utils
# MAGIC %run ../../../../utils/logging

# COMMAND ----------

# DBTITLE 1,Spark utils
# MAGIC %run ../../../../utils/spark

# COMMAND ----------

# DBTITLE 1,Delta utils
# MAGIC %run ../../../../utils/delta

# COMMAND ----------

# DBTITLE 1,Parameters
MAIN_DIR: str = dbutils.widgets.get("main_dir")              # base folder with FEATURES
EXECUTION_DATE: str = dbutils.widgets.get("execution_date")  # YYYYMMDD (a Sunday)
MODEL_URI: str = dbutils.widgets.get("model_uri")            # MLflow model URI
ENVIRONMENT: str = "prd" if dbutils.secrets.get("KV", "ENV") == "PRD" else "dev"

# COMMAND ----------

# DBTITLE 1,Input path
FEATURES_PATH = os.path.join(MAIN_DIR, "FEATURES")

# COMMAND ----------

# DBTITLE 1,Output path
PREDICTIONS_PATH = f"models_{ENVIRONMENT}.mx.da_predictions"

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
ENTITY_MAP_COL = "entity_map"
TIMESTAMP_COL = "ldts"

# COMMAND ----------

# DBTITLE 1,Resolve calendar values
predicted_week = resolve_calendar_value(EXECUTION_DATE, "iso_year_week_1_d")
current_month = resolve_calendar_value(EXECUTION_DATE, "calmonth")

log(f"Predicting week: {predicted_week} (execution_date={EXECUTION_DATE})", "INFO")

# COMMAND ----------

# DBTITLE 1,Load features table
# Read features and filter to target week
features_df: DataFrame = spark.read.load(FEATURES_PATH).filter(F.col(WEEK_COL) == F.lit(predicted_week))

# Existence check
if len(features_df.head(1)) == 0:
    raise RuntimeError(f"No features found for week {predicted_week} in {FEATURES_PATH}")

log(f"Loaded features DataFrame with {len(features_df.columns)} columns.", "INFO")

# COMMAND ----------

# DBTITLE 1,Create MLflow UDF and score
# Configure MLflow
mlflow.set_registry_uri("databricks-uc")
ml_client = MlflowClient()

# Build struct containing feature columns
scoring_struct = F.struct(*[F.col(c) for c in features_df.columns])

# Create pyfunc spark UDF
predict_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=MODEL_URI,
    result_type="double",
    env_manager="virtualenv",
)

# Apply scoring
scored_df = features_df.withColumn(PREDICTION_COL, predict_udf(scoring_struct))

# COMMAND ----------

# DBTITLE 1,Build entity map and register ID
# Build entity mapping
entity_mapping = {
    CUSTOMER_ID_COL: F.col(CUSTOMER_ID_COL),
    BPR_COL: F.col(BPR_COL),
    WEEK_COL: F.col(WEEK_COL),
}

# Compose deterministic record ID expression
scored_df = compose_record_id(
    scored_df,
    [CUSTOMER_ID_COL, BPR_COL, WEEK_COL],
    MODEL_URI,
    RECORD_ID_COL,
)

# Create entity map
scored_df = create_entity_map_column(
    scored_df, entity_mapping, ENTITY_MAP_COL
)

# COMMAND ----------

# DBTITLE 1,Build final output
predictions_df = scored_df.select(
    F.col(RECORD_ID_COL),
    F.col(PREDICTION_COL),
    F.col(ENTITY_MAP_COL),
    F.lit(BUSINESS_UNIT_NAME).alias(BUSINESS_UNIT_COL),
    F.lit(MODEL_URI).alias(MODEL_URI_COL),
    F.lit(current_month).alias(CALMONTH_COL),
    F.current_timestamp().alias(TIMESTAMP_COL),
)

# COMMAND ----------

# DBTITLE 1,Persist predictions
# Upsert to metastore
delta_upsert(
    source_df=predictions_df,
    target_identifier=PREDICTIONS_PATH,
    merge_keys=[RECORD_ID_COL],
    partition_col_or_cols=[MODEL_URI_COL, CALMONTH_COL],
    as_metastore=True,
)

log(f"Predictions upserted to table {PREDICTIONS_PATH} (week={predicted_week}).", "INFO")
