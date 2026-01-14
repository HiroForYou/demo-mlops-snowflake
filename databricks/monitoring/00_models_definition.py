# Databricks notebook source
# MAGIC %md
# MAGIC # Models definition for feature drift
# MAGIC
# MAGIC **Objective**: This notebook defines (collect and return) the model URIs used for inference during the current week and return them as an output list.
# MAGIC
# MAGIC Input widgets:
# MAGIC - EXECUTION_DATE: Execution date of module, must be a sunday in yyyyMMdd format. <br>
# MAGIC Ex: 20250831 corresponds to August 31st, 2025. (A sunday).
# MAGIC
# MAGIC Input:
# MAGIC - Predictions data table
# MAGIC
# MAGIC Output:
# MAGIC - models_list: list of model URIs that identify models in Unity Catalog, in json format. <br>
# MAGIC Ex. ["models:/models_prd.mx.forecast_bpr_customer_week@20250819"].
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Import packages
import json
import pyspark.sql.functions as sf

# COMMAND ----------

# DBTITLE 1,Load calendar functions
# MAGIC %run ../../../../utils/calendar

# COMMAND ----------

# DBTITLE 1,Widgets
EXECUTION_DATE = dbutils.widgets.get("execution_date")

# COMMAND ----------

# DBTITLE 1,Domain column names
MODEL_URI_COL = "model_uri"
WEEK_COL = "week"
PREDICTION_COL = "prediction"
ENTITY_MAP_COL = "entity_map"
TIMESTAMP_COL = "ldts"

# COMMAND ----------

# DBTITLE 1,Input path
PREDICTIONS_TABLE = "models_prd.mx.da_predictions"

# COMMAND ----------

# DBTITLE 1,Load calendar parameters
execution_week = resolve_calendar_value(EXECUTION_DATE, "iso_year_week_1_d")

if not execution_week:
    raise RuntimeError(f"Execution date {EXECUTION_DATE} not found in calendar table.")

# COMMAND ----------

# DBTITLE 1,Filter predictions table with current week
current_week_preds = (
    spark.table(PREDICTIONS_TABLE)
    .filter(
        (sf.col(ENTITY_MAP_COL)[WEEK_COL] == sf.lit(execution_week))
        & (sf.col(MODEL_URI_COL).contains("forecast_bpr_customer_week"))
    )
    .select(
        MODEL_URI_COL,
        PREDICTION_COL,
        TIMESTAMP_COL,
        ENTITY_MAP_COL,  # To obtain "week"
    )
)

if len(current_week_preds.take(1)) == 0:
    raise RuntimeError(f"No predictions found for week {EXECUTION_DATE}. Exiting.")

# COMMAND ----------

# DBTITLE 1,Collect inference models in current week
models_list = [
    r[MODEL_URI_COL]
    for r in current_week_preds.select(MODEL_URI_COL).distinct().collect()
]

# COMMAND ----------

# DBTITLE 1,Exit model ids
# Exit list of model aliases
dbutils.notebook.exit(json.dumps(models_list))
