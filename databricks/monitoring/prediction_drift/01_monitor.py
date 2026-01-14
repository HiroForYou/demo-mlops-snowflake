# Databricks notebook source
# MAGIC %md
# MAGIC # Univariate Prediction Drift Report
# MAGIC
# MAGIC **Objective**: This notebook estimates univariate drift scores for the predictions of Suggested Order models.
# MAGIC
# MAGIC Input widgets:
# MAGIC - MAIN_DIR: DL directory which stores features, predictions and previous drift scores. Should default to `/mnt/DL/PROCESSED/MODELS_UC/`
# MAGIC - EXECUTION_DATE: Execution date of module, must be a sunday in yyyyMMdd format. Ex: 20250831 corresponds to August 31st, 2025. (A sunday).
# MAGIC - MODEL_URI: MLflow model URI, Ex: `models:/models_prd.mx.forecast_bpr_customer_week@20250910`.
# MAGIC
# MAGIC Inputs:
# MAGIC - Predictions and Inference data
# MAGIC
# MAGIC Outputs:
# MAGIC - Univariate prediction drift report
# MAGIC
# MAGIC Process:
# MAGIC 1. Fetch UC monitored model
# MAGIC 2. Determine path to train dataset
# MAGIC 3. Join all train datasets and filter per group (reference set)
# MAGIC 4. Sample current week features data and filter per group (analysis set)
# MAGIC 5. Launch and fit a `UnivariateDriftCalculator` object per group
# MAGIC 6. Calculate final report
# MAGIC 7. Add metadata and write to disk

# COMMAND ----------

# DBTITLE 1,Import packages
import os
import datetime

import mlflow
from mlflow.tracking import MlflowClient

import pyspark.sql.functions as sf

# COMMAND ----------

# DBTITLE 1,Load univariate drift calculator
# MAGIC %run ../../../../../advanced_analytics/advanced_analytics/drift/univariate/calculator

# COMMAND ----------

# DBTITLE 1,Load delta upsert function
# MAGIC %run ../../../../../utils/delta

# COMMAND ----------

# DBTITLE 1,Load spark functions
# MAGIC %run ../../../../../utils/spark

# COMMAND ----------

# DBTITLE 1,Load calendar functions
# MAGIC %run ../../../../../utils/calendar

# COMMAND ----------

# DBTITLE 1,MLFlow configuration
mlflow.set_registry_uri("databricks-uc")
ml_client = MlflowClient()

# COMMAND ----------

# DBTITLE 1,Widgets
MAIN_DIR = dbutils.widgets.get("main_dir")
MODEL_URI = dbutils.widgets.get("model_uri")
EXECUTION_DATE = dbutils.widgets.get("execution_date")

# COMMAND ----------

# DBTITLE 1,Parameters
# Model parameters
MODEL_TYPE = MODEL_URI.split("@")[0].split(".")[2].split("_")[0]
BU = MODEL_URI.split("@")[0].split(".")[1]

# Environment parameters
ENV = "prd" if (dbutils.secrets.get("KV", "ENV") == "PRD") else "dev"

# COMMAND ----------

# DBTITLE 1,Get specifications per model
# Define group column and thresholds per model type
FORECAST_DICT = {
    "group_col": "stats_ntile_group",
    "thresholds": {"jensen_shannon": (0, 0.2, 0.45)},
}

PROPENSITY_DICT = {
    "group_col": "group",
    "thresholds": {"jensen_shannon": (0, 0.2, 0.45)},
}

# Identify model type and get the corresponding dictionary
MODEL_TYPE_DICT = {
    "forecast": FORECAST_DICT,
    "propensity": PROPENSITY_DICT,
}
MODEL_DICT = MODEL_TYPE_DICT.get(MODEL_TYPE, FORECAST_DICT)

# COMMAND ----------

# DBTITLE 1,Column parameters
# Dictionary for identify bkcc column with bu
BKCC_DICT = {
    "mx": "MXBEB",
    "pe": "PEBEB",
    "ar": "ARBEB",
    "ccswb": "USBEB",
    "ec": "ECBEB",
}

# Column names
RECORD_ID_COL = "record_id"
MODEL_URI_COL = "model_uri"
GROUP_COL_NAME = MODEL_DICT["group_col"]
PREDICTION_COL = "prediction"
METHOD_COL = "method"
COLUMN_COL = "column"
VALUE_COL = "value"
METRIC_NAME_COL = "metric_name"
METRIC_VALUE_COL = "metric_value"
AGGREGATED_COL = "aggregated_col"
AGGREGATED_VALUE_COL = "aggregated_value"
ALERT_COL = "alert"
ALERT_LEVEL_COL = "alert_level"
LOWER_COL = "lower_threshold"
WARNING_COL = "warning_threshold"
CRITICAL_COL = "critical_threshold"
WEEK_COL = "week"
CALMONTH_COL = "calmonth"
BUSINESS_UNIT_COL = "bkcc"
ENTITY_MAP_COL = "entity_map"
TIMESTAMP_COL = "ldts"

# Algorithm constants
IN_CONTROL_WEEKS = 14  # weeks to use for baseline (from cutoff_date)
TZ = "America/Monterrey"  # timezone for timestamps

# COMMAND ----------

# DBTITLE 1,Paths
# Input
CALENDAR_PATH = "prd.gold.v_rt_calendar"
FEATURES_PATH = os.path.join(
    MAIN_DIR, MODEL_URI.split("@")[0].split(".")[2].upper(), "FEATURES"
)
PREDICTIONS_TABLE = f"models_prd.{BU}.da_predictions"

# Output
DRIFT_REPORT_TABLE = f"models_{ENV}.{BU}.da_prediction_drift"

# COMMAND ----------

# DBTITLE 1,Load calendar parameters
calendar_df = (
    spark.table(CALENDAR_PATH)
    .select(
        F.col("date_d").cast("date").alias("date_d"),
        F.col("iso_year_week_d").alias("iso_week"),
    )
    .distinct()
)

# Get current ISO week
current_preds_iso_year_week = resolve_calendar_value(EXECUTION_DATE, "iso_year_week_1_d")

# Get current month
current_month = resolve_calendar_value(EXECUTION_DATE, "calmonth")

# COMMAND ----------

# DBTITLE 1,Define cadidate weeks for baseline
# List of tuples (model_uri, week) for all baseline weeks
baseline_model_week_rows = []

try:
    # Validate model URI and extract alias
    if not MODEL_URI.startswith("models:/"):
        raise ValueError(f"MODEL_URI not models:/ URI: {MODEL_URI}")
    payload = MODEL_URI[len("models:/") :]
    if "@" not in payload:
        raise ValueError(f"MODEL_URI missing alias: {MODEL_URI}")
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
            run_tags = dict(run.data.tags) if run and run.data and run.data.tags else {}
            cutoff_str = run_tags.get("cutoff_date")

    if not cutoff_str:
        raise RuntimeError(f"cutoff_date tag not found for model {MODEL_URI}")

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
        raise RuntimeError(f"No weeks found in calendar after cutoff {cutoff_str}")

    # Append rows (week) for all weeks
    for w in weeks_list:
        baseline_model_week_rows.append(w)

except Exception as e:
    raise RuntimeError(f"Skipping model {MODEL_URI} due to: {e}")

if not baseline_model_week_rows:
    raise RuntimeError(f"No baseline weeks resolved for model {MODEL_URI}. Aborting.")

# COMMAND ----------

# DBTITLE 1,Prepare dimension tables
# Load model predictions
predictions_df = spark.table(PREDICTIONS_TABLE)

# Load inference data
inference_data_df = (
    spark.read.load(FEATURES_PATH)
    .filter(sf.col(WEEK_COL) == current_preds_iso_year_week)
    .select(sf.col("customer_id").cast("int"), "brand_pres_ret", GROUP_COL_NAME)
)

# COMMAND ----------

# DBTITLE 1,Prepare base dataset
prediction_drift_set = (
    predictions_df.filter(sf.col(MODEL_URI_COL) == MODEL_URI)
    .filter(
        sf.col(ENTITY_MAP_COL)[WEEK_COL].isin(
            baseline_model_week_rows + [current_preds_iso_year_week]
        )
    )
    .join(
        inference_data_df,
        [
            sf.col(ENTITY_MAP_COL)["customer_id"] == sf.col("customer_id"),
            sf.col(ENTITY_MAP_COL)["brand_pres_ret"] == sf.col("brand_pres_ret"),
        ],
        "inner",
    )
    .select(*[sf.col(c) for c in predictions_df.columns], sf.col(GROUP_COL_NAME))
)

# COMMAND ----------

# DBTITLE 1,Combine and Filter Training Datasets
# Get the group per model to get a reference set per each one
groups = [
    row[GROUP_COL_NAME]
    for row in prediction_drift_set.select(GROUP_COL_NAME).distinct().collect()
]

# Create a dictionary with keys to the reference set per group
reference_sets_per_group = {
    group: prediction_drift_set.filter(
        (sf.col(GROUP_COL_NAME) == group)
        & (sf.col(ENTITY_MAP_COL)[WEEK_COL] != current_preds_iso_year_week)
    )
    for group in groups
}

# COMMAND ----------

# DBTITLE 1,Filter Current Features Set
# Use historical data of the current week in features divided per group
analysis_sets_per_group = {
    group: prediction_drift_set.filter(
        (sf.col(GROUP_COL_NAME) == group)
        & (sf.col(ENTITY_MAP_COL)[WEEK_COL] == current_preds_iso_year_week)
    )
    for group in groups
}

# COMMAND ----------

# DBTITLE 1,Verify all reference sets contain data
# Verify that all groups in reference_sets_per_group contain data
for group, df in reference_sets_per_group.items():
    if df.limit(1).count() == 0:
        raise Exception(f"Reference set for {group} has no data for baseline ({baseline_model_week_rows} weeks).")

# COMMAND ----------

# DBTITLE 1,Verify all analysis sets contain data
# Verify that all groups in analysis_sets_per_group contain data
for group, df in analysis_sets_per_group.items():
    if df.limit(1).count() == 0:
        raise Exception(f"Analysis set {group} has no data for current week {current_preds_iso_year_week}.")

# COMMAND ----------

# DBTITLE 1,Identify and classify drift columns
# Drift column is only the prediction column
drift_columns = [PREDICTION_COL]

# Classify prediction as numerical or categorial
cols_classification = classify_columns(
    df=prediction_drift_set.select(PREDICTION_COL), classify_for="prediction drift"
)

categorical_cols = (
    [PREDICTION_COL]
    if PREDICTION_COL in cols_classification["categorical_cols"]
    else []
)
numerical_cols = (
    [PREDICTION_COL] if PREDICTION_COL in cols_classification["numerical_cols"] else []
)

# COMMAND ----------

# DBTITLE 1,Create the UnivariateDriftCalculator object
# Configuration of the drift calculator
uni_drift_calc_obj = UnivariateDriftCalculator(
    categorical_methods=["jensen_shannon"],
    continuous_methods=["jensen_shannon"],
    treat_as_categorical=categorical_cols,
    column_names=drift_columns,
    thresholds=MODEL_DICT["thresholds"],
)

# COMMAND ----------

# DBTITLE 1,Calculate Univariate Drift per Group
final_uni_drift_report = None

for group in groups:
    # Fit the calculator
    uni_drift_calc = uni_drift_calc_obj.fit(reference_sets_per_group[group])
    # Calculate drift
    uni_drift_report = uni_drift_calc.calculate(analysis_sets_per_group[group])

    # Add aggregated_col to identify the aggrupation column used
    uni_drift_report = uni_drift_report.withColumn(
        AGGREGATED_COL, sf.lit(MODEL_DICT["group_col"])
    )

    # Add aggregated_value to identify the report's group
    uni_drift_report = uni_drift_report.withColumn(AGGREGATED_VALUE_COL, sf.lit(group))

    uni_drift_report = uni_drift_report.withColumn(
        TIMESTAMP_COL,
        sf.from_utc_timestamp(sf.current_timestamp(), TZ),  # Add current timestamp
    )

    if final_uni_drift_report is None:
        final_uni_drift_report = uni_drift_report
    else:
        final_uni_drift_report = final_uni_drift_report.unionByName(uni_drift_report)

# COMMAND ----------

# DBTITLE 1,Add week, bkcc and partition columns to report
final_uni_drift_report = (
    final_uni_drift_report.withColumn(
        # To identify the time period analyzed
        WEEK_COL, sf.lit(current_preds_iso_year_week),
    )
    .withColumn(
        # To identify the business coalision
        BUSINESS_UNIT_COL, sf.lit(BKCC_DICT[BU]),
    )
    .withColumn(MODEL_URI_COL, sf.lit(MODEL_URI))           # Partition by model id
    .withColumn(CALMONTH_COL, sf.lit(current_month))        # Partition by month
)

# COMMAND ----------

# DBTITLE 1,Join drift report with thresholds and align schema
thresholds_df = spark.createDataFrame(
    [
        (method, lower, warning, critical)
        for method, (lower, warning, critical) in MODEL_DICT["thresholds"].items()
    ],
    [METHOD_COL, LOWER_COL, WARNING_COL, CRITICAL_COL],
)

final_uni_drift_report = final_uni_drift_report.join(
    other=thresholds_df, on=METHOD_COL
).select(
    sf.col(MODEL_URI_COL),
    sf.col(WEEK_COL),
    sf.col(BUSINESS_UNIT_COL),
    sf.col(METHOD_COL).alias(METRIC_NAME_COL),
    sf.col(VALUE_COL).alias(METRIC_VALUE_COL),
    sf.col(AGGREGATED_COL),
    sf.col(AGGREGATED_VALUE_COL),
    WARNING_COL,
    CRITICAL_COL,
    sf.col(ALERT_COL).alias(ALERT_LEVEL_COL),
    sf.col(CALMONTH_COL),
    sf.col(TIMESTAMP_COL),
)

# COMMAND ----------

# DBTITLE 1,Build entity map
# Build entity_map
entity_mapping = {
    WEEK_COL: F.lit(current_preds_iso_year_week),
}
final_uni_drift_report = create_entity_map_column(
    final_uni_drift_report, entity_mapping, ENTITY_MAP_COL
)

# COMMAND ----------

# DBTITLE 1,Compose a deterministic record id per row
# Add record_id column
final_uni_drift_report = compose_record_id(
    final_uni_drift_report,
    [WEEK_COL, AGGREGATED_COL, AGGREGATED_VALUE_COL, METRIC_NAME_COL],
    MODEL_URI_COL,
    RECORD_ID_COL,
)

# COMMAND ----------

# DBTITLE 1,Define final report to save
# final_uni_drift_report schema
# [record_id, model_uri, week, bkcc, metric_name, column, metric_value, ldts, warning_threshold, critical_threshold, alert, group_col, calmonth, entity_map]

# Select columns to be saved
report = final_uni_drift_report.select(
    sf.col(RECORD_ID_COL),
    sf.col(MODEL_URI_COL),
    sf.col(ENTITY_MAP_COL),
    sf.col(AGGREGATED_COL),
    sf.col(AGGREGATED_VALUE_COL),
    sf.col(METRIC_NAME_COL),
    sf.col(METRIC_VALUE_COL),
    sf.col(WARNING_COL),
    sf.col(CRITICAL_COL),
    sf.col(ALERT_LEVEL_COL),
    sf.col(BUSINESS_UNIT_COL),
    sf.col(CALMONTH_COL),
    sf.col(TIMESTAMP_COL),
)

# COMMAND ----------

# DBTITLE 1,Save final drift report
# Use delta_upset function for report writing in UC
delta_upsert(
    source_df=report,
    target_identifier=DRIFT_REPORT_TABLE,
    merge_keys=[RECORD_ID_COL],
    partition_col_or_cols=[
        MODEL_URI_COL,
        CALMONTH_COL,
    ],
    as_metastore=True,
)
