# Databricks notebook source
# MAGIC %md
# MAGIC # Performance Drift Detection
# MAGIC
# MAGIC **Description:** This notebooks evaluates the performance drift of forecasting models by comparing the metrics observed in the current week against a baseline per model constructed from the weeks following the `cutoff_date` recorder in MLflow. It produces alert signals (OK / WARNING / CRITICAL / NO_BASELINE) written to a Delta table for consumption by dashboards and runbooks.
# MAGIC
# MAGIC ## Inputs
# MAGIC
# MAGIC **Execution Parameters:**
# MAGIC * `execution_date` (string): reference date in a format accepted by `resolve_calendar_value`.
# MAGIC
# MAGIC **Tables and Services**
# MAGIC * `CALENDAR_PATH` (Delta): Calendar table.
# MAGIC * `METRICS_PATH` (Delta): Table with historical metrics.
# MAGIC * **MLflow registry access**: This notebook queries model versions by alias using `MLflowClient.get_model_version_by_alias`.
# MAGIC
# MAGIC **Algorithm Constants**
# MAGIC * `IN_CONTROL_WEEKS` (int): Number of weeks to consider from `cutoff_date` to construct the baseline.
# MAGIC * `WARNING_SIGMA` (float): Standard deviation multiplier for the WARNING threshold.
# MAGIC * `CRITICAL_SIGMA` (float): Standard deviation multiplier for the CRITICAL threshold.
# MAGIC * `STD_EPS` (float): Numeric epsilon to avoid stddev = 0.
# MAGIC * `TZ` (string): Time zone used for `ldts` (e.g., `America/Monterrey`).
# MAGIC
# MAGIC ## Outputs
# MAGIC * `METRICS_PATH` (upsert): Performance drift alerts

# COMMAND ----------

# DBTITLE 1,Import dependencies
import os
import datetime
from typing import List, Tuple

import pyspark.sql.functions as F
from pyspark.sql import types as T
from pyspark.sql import DataFrame

import mlflow
from mlflow.tracking import MlflowClient

# COMMAND ----------

# DBTITLE 1,Calendar utilities
# MAGIC %run ../../../../../utils/calendar

# COMMAND ----------

# DBTITLE 1,Delta utilities
# MAGIC %run ../../../../../utils/delta

# COMMAND ----------

# DBTITLE 1,MLflow configuration
mlflow.set_registry_uri("databricks-uc")
ml_client = MlflowClient()

# COMMAND ----------

# DBTITLE 1,Execution parameters
EXECUTION_DATE = dbutils.widgets.get("execution_date")
ENVIRONMENT = "prd" if dbutils.secrets.get("KV", "ENV") == "PRD" else "dev"

# COMMAND ----------

# DBTITLE 1,Input paths
CALENDAR_PATH = "prd.gold.v_rt_calendar"
METRICS_PATH = f"models_{ENVIRONMENT}.mx.da_performance_drift"

# COMMAND ----------

# DBTITLE 1,Algorithm constants
IN_CONTROL_WEEKS = 15  # weeks to use from cutoff_date for baseline
WARNING_SIGMA = 2.0  # stddev multiplier for WARNING
CRITICAL_SIGMA = 3.0  # stddev multiplier for CRITICAL
STD_EPS = 1e-9  # epsilon for safe stddev (prevents division by zero)
TZ = "America/Monterrey"  # timezone for timestamps

# COMMAND ----------

# DBTITLE 1,Domain column names
RECORD_ID_COL = "record_id"
MODEL_URI_COL = "model_uri"
CALMONTH_COL = "calmonth"
WEEK_COL = "week"
ENTITY_MAP_COL = "entity_map"
TIMESTAMP_COL = "ldts"
AGGREGATED_COL = "aggregated_col"
AGGREGATED_VALUE_COL = "aggregated_value"
METRIC_NAME_COL = "metric_name"
METRIC_VALUE_COL = "metric_value"
WARNING_COL = "warning_threshold"
CRITICAL_COL = "critical_threshold"
ALERT_COL = "alert_level"

# COMMAND ----------

# DBTITLE 1,Load calendar parameters
calendar_df = (
    spark.table(CALENDAR_PATH)
    .select(
        F.col("date_d").cast("date").alias("date_d"),
        F.col("iso_year_week_1_d").alias("iso_week"),
    )
    .distinct()
)

inference_week = compute_x_days_ago_value(EXECUTION_DATE, days_ago=7, return_col="iso_year_week_d")

# COMMAND ----------

# DBTITLE 1,Load metrics
metrics_df = (
    spark.read.format("delta")
    .table(METRICS_PATH)
    .select(
        RECORD_ID_COL,
        MODEL_URI_COL,
        ENTITY_MAP_COL,
        F.col(ENTITY_MAP_COL).getItem(WEEK_COL).alias(WEEK_COL),
        AGGREGATED_COL,
        AGGREGATED_VALUE_COL,
        METRIC_NAME_COL,
        F.col(METRIC_VALUE_COL).cast("double").alias(METRIC_VALUE_COL),
        CALMONTH_COL,
    )
    .filter(F.col(WEEK_COL) <= F.lit(inference_week))
)
current_metrics = metrics_df.filter(
    (F.col(WEEK_COL) == inference_week)
    & F.col(MODEL_URI_COL).contains("forecast_bpr_customer_week")
)

if len(current_metrics.take(1)) == 0:
    raise RuntimeError(f"No metrics found for week {inference_week}. Exiting.")

# COMMAND ----------

# DBTITLE 1,Construct baseline weeks
model_list = [
    r[MODEL_URI_COL] for r in current_metrics.select(MODEL_URI_COL).distinct().collect()
]

# List of tuples (model_uri, week) for all baseline weeks
baseline_model_week_rows: List[Tuple[str, str]] = []

for model_uri in model_list:
    try:
        # Validate model URI and extract alias
        if not model_uri.startswith("models:/"):
            raise ValueError(f"model_uri not models:/ URI: {model_uri}")
        payload = model_uri[len("models:/") :]
        if "@" not in payload:
            raise ValueError(f"model_uri missing alias: {model_uri}")
        model_name, model_alias = payload.split("@", 1)

        # Resolve model version by alias
        mv = ml_client.get_model_version_by_alias(model_name, model_alias)
        tags = getattr(mv, "tags", None) or {}
        cutoff_str = tags.get("cutoff_date")
        if not cutoff_str:
            # Fallback to run tags if model-version tags missing
            run_id = getattr(mv, "run_id", None)
            if run_id:
                run = ml_client.get_run(run_id)
                run_tags = (
                    dict(run.data.tags) if run and run.data and run.data.tags else {}
                )
                cutoff_str = run_tags.get("cutoff_date")

        if not cutoff_str:
            raise RuntimeError(f"cutoff_date tag not found for model {model_uri}")

        # Validate and convert cutoff_date (string in 'YYYYMMDD' format) into a date object.
        # This represents the last date included in the model's training window (the cutoff boundary).
        cutoff_date = datetime.datetime.strptime(cutoff_str, "%Y%m%d").date()

        # Compute the date window for the  "in-control" evaluation period:
        # it spans from the model's cutoff_date (training boundary) up to IN_CONTROL_WEEKS weeks later
        start_date = cutoff_date
        end_date = (
            cutoff_date
            + datetime.timedelta(weeks=IN_CONTROL_WEEKS)
            - datetime.timedelta(days=1)
        )

        # Query calendar_df for weeks in that date interval
        weeks_rows = (
            calendar_df.filter(
                (F.col("date_d") >= F.lit(start_date))
                & (F.col("date_d") <= F.lit(end_date))
            )
            .select("iso_week")
            .distinct()
            .collect()
        )
        weeks_list = sorted([r["iso_week"] for r in weeks_rows])

        if not weeks_list:
            continue

        # append rows (model_uri, week) for all weeks
        for w in weeks_list:
            baseline_model_week_rows.append((model_uri, w))

    except Exception as e:
        continue

if not baseline_model_week_rows:
    raise RuntimeError("No baseline weeks resolved for any model. Aborting.")

# COMMAND ----------

# DBTITLE 1,Compute baseline stats
# Create DataFrame (model_uri, week) describing all baselines
schema = T.StructType(
    [
        T.StructField(MODEL_URI_COL, T.StringType(), False),
        T.StructField(WEEK_COL, T.StringType(), False),
    ]
)
baseline_model_week_df = spark.createDataFrame(baseline_model_week_rows, schema=schema)
baseline_model_week_df = F.broadcast(baseline_model_week_df)

# Compute baseline stats
baseline_stats = (
    metrics_df.join(baseline_model_week_df, on=[MODEL_URI_COL, WEEK_COL], how="inner")
    .groupBy(MODEL_URI_COL, AGGREGATED_COL, AGGREGATED_VALUE_COL, METRIC_NAME_COL)
    .agg(
        F.countDistinct(WEEK_COL).alias("n_baseline_weeks"),
        F.mean(METRIC_VALUE_COL).cast("double").alias("mean_baseline"),
        F.stddev(METRIC_VALUE_COL).cast("double").alias("stddev_baseline"),
    )
)

# COMMAND ----------

# DBTITLE 1,Generate alerts DataFrame
# Compute a sign that encoded directionality:
#   sign = +1.0 for metrics where higher == worse (e.g., WAPE)
#   sign = -1.0 for metrics where lower == worse (e.g., F1)
sign_expr = (
    F.when(F.col(METRIC_NAME_COL) == "WAPE", F.lit(1.0))
    .when(F.col(METRIC_NAME_COL) == "F1", F.lit(-1.0))
    .otherwise(F.lit(1.0))  # default: higher is worse
)

# Compute a null-safe stddev expression
safe_stddev_expr = F.greatest(
    F.coalesce(F.col("stddev_baseline"), F.lit(0.0)), F.lit(float(STD_EPS))
)

# Join current-week metrics with baseline stats
alerts_df = (
    current_metrics.join(
        baseline_stats,
        on=[MODEL_URI_COL, AGGREGATED_COL, AGGREGATED_VALUE_COL, METRIC_NAME_COL],
        how="left",
    )
    .withColumn("safe_stddev", safe_stddev_expr)
    .withColumn("direction_sign", sign_expr)
    .withColumn(
        # signed difference: positive when 'worse' (uniform across metric types)
        "signed_diff",
        F.col("direction_sign") * (F.col(METRIC_VALUE_COL) - F.col("mean_baseline")),
    )
    .withColumn(
        # warning and critical thresholds: algebraic form mean + sign * k * stddev
        WARNING_COL,
        F.col("mean_baseline")
        + F.col("direction_sign") * F.lit(float(WARNING_SIGMA)) * F.col("safe_stddev"),
    )
    .withColumn(
        CRITICAL_COL,
        F.col("mean_baseline")
        + F.col("direction_sign") * F.lit(float(CRITICAL_SIGMA)) * F.col("safe_stddev"),
    )
    .withColumn(
        # baseline is valid if we have a mean and at least one baseline week
        "baseline_valid",
        (F.col("n_baseline_weeks").isNotNull())
        & (F.col("n_baseline_weeks") > 0)
        & F.col("mean_baseline").isNotNull(),
    )
    .withColumn(
        # alert_level mapping: -1=no_baseline, 2=critical, 1=warning, 0=ok
        ALERT_COL,
        F.when(~F.col("baseline_valid"), F.lit(-1))
        .when(
            F.col("signed_diff") >= F.lit(float(CRITICAL_SIGMA)) * F.col("safe_stddev"),
            F.lit(2),
        )
        .when(
            F.col("signed_diff") >= F.lit(float(WARNING_SIGMA)) * F.col("safe_stddev"),
            F.lit(1),
        )
        .otherwise(F.lit(0)),
    )
    .select(
        RECORD_ID_COL,
        MODEL_URI_COL,
        ENTITY_MAP_COL,
        AGGREGATED_COL,
        AGGREGATED_VALUE_COL,
        METRIC_NAME_COL,
        METRIC_VALUE_COL,
        WARNING_COL,
        CRITICAL_COL,
        ALERT_COL,
        CALMONTH_COL,
    )
    .withColumn(TIMESTAMP_COL, F.from_utc_timestamp(F.current_timestamp(), TZ))
)

# COMMAND ----------

# DBTITLE 1,Upsert alerts
delta_upsert(
    source_df=alerts_df,
    target_identifier=METRICS_PATH,
    merge_keys=[RECORD_ID_COL],
    partition_col_or_cols=[MODEL_URI_COL, CALMONTH_COL],
    as_metastore=True,
)
