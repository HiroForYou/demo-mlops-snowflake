# Databricks notebook source
# MAGIC %md
# MAGIC # Univariate Feature Drift Report
# MAGIC
# MAGIC **Objective**: This notebook estimates univariate drift scores for a set of input feature columns for the Suggested Order models.
# MAGIC
# MAGIC Input widgets:
# MAGIC - MAIN_DIR: DL directory which stores features, predictions and previous drift scores. Should default to `/mnt/DL/PROCESSED/MODELS_UC/`
# MAGIC - EXECUTION_DATE: Execution date of module, must be a sunday in yyyyMMdd format. Ex: 20250831 corresponds to August 31st, 2025. (A sunday).
# MAGIC - MODEL_URI: MLflow model URI, Ex: `models:/models_prd.mx.forecast_bpr_customer_week@20250910`.
# MAGIC
# MAGIC Inputs:
# MAGIC - Features
# MAGIC - Historical training data for the monitored model
# MAGIC
# MAGIC Outputs:
# MAGIC - Univariate data drift report
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
import pyspark.sql.functions as sf
from functools import reduce

import mlflow

mlflow.set_registry_uri("databricks-uc")

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

# DBTITLE 1,Domain column parameters
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
METHOD_COL = "method"
COLUMN_COL = "column"
VALUE_COL = "value"
FEATURE_NAME_COL = "feature_name"
AGGREGATED_COL = "aggregated_col"
AGGREGATED_VALUE_COL = "aggregated_value"
METRIC_NAME_COL = "metric_name"
METRIC_VALUE_COL = "metric_value"
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
TZ = "America/Monterrey"  # timezone for timestamps

# COMMAND ----------

# DBTITLE 1,Paths
# Input
FEATURES_PATH = os.path.join(
    MAIN_DIR, MODEL_URI.split("@")[0].split(".")[2].upper(), "FEATURES"
)

# Output
DRIFT_REPORT_TABLE = f"models_{ENV}.{BU}.da_data_drift"

# COMMAND ----------

# DBTITLE 1,Load Pyfunc based model
model = mlflow.pyfunc.spark_udf(
    spark, model_uri=MODEL_URI, env_manager="virtualenv"
)  # Adapt to RT
train_set_path = model.metadata.metadata.get("train_set_path")

# COMMAND ----------

# DBTITLE 1,Load calendar parameters
# Get current ISO week
current_features_iso_year_week = resolve_calendar_value(
    EXECUTION_DATE, "iso_year_week_1_d"
)

# Get current month
current_month = resolve_calendar_value(EXECUTION_DATE, CALMONTH_COL)

# COMMAND ----------

# DBTITLE 1,Base data source
# For feature drift filtered by target week (week of execution)
current_week_features = spark.read.load(FEATURES_PATH).filter(
    (sf.col(WEEK_COL) == current_features_iso_year_week)
)

# COMMAND ----------

# DBTITLE 1,Combine and filter training datasets
paths = [f.path.lstrip("dbfs:") for f in dbutils.fs.ls(train_set_path)]
datasets = [spark.read.load(path) for path in paths]

# Join all the datasets into one
training_sets = reduce(lambda x, y: x.unionByName(y), datasets)

# Get the group per model to get a reference set per each one
groups = [
    row[GROUP_COL_NAME]
    for row in training_sets.select(GROUP_COL_NAME).distinct().collect()
]

# Create a dictionary with keys to the reference set per group
reference_sets_per_group = {
    group: training_sets.filter((sf.col(GROUP_COL_NAME) == group)) for group in groups
}

# COMMAND ----------

# DBTITLE 1,Filter current features set
# Use historical data of the current week in features divided per group
analysis_sets_per_group = {
    group: current_week_features.filter(sf.col(GROUP_COL_NAME) == group)
    for group in groups
}

# COMMAND ----------

# DBTITLE 1,Verify all analysis sets contain data
# Verify that all groups in analysis_sets_per_group contain data
for group, df in analysis_sets_per_group.items():
    if df.limit(1).count() == 0:
        raise Exception(
            f"{group} has no data for week {current_features_iso_year_week}."
        )

# COMMAND ----------

# DBTITLE 1,Identify drift columns
# Classify columns into excluded, categorical and numerical
cols_classification = classify_columns(next(iter(reference_sets_per_group.values())))

drift_columns = [
    c
    for c in next(iter(reference_sets_per_group.values())).columns
    if c not in cols_classification["exclude_cols"]
]
categorical_cols = cols_classification["categorical_cols"]
numerical_cols = [
    c for c in drift_columns if c not in cols_classification["categorical_cols"]
]

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

# DBTITLE 1,Calculate univariate drift per group
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
        WEEK_COL, sf.lit(current_features_iso_year_week)
    )  
    .withColumn(
        # To identify the business coalision
        BUSINESS_UNIT_COL, sf.lit(BKCC_DICT[BU])
    )  
    .withColumn(MODEL_URI_COL, sf.lit(MODEL_URI))           # Partition by model id
    .withColumn(CALMONTH_COL, sf.lit(current_month))           # Partition by month
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
    sf.col(COLUMN_COL).alias(FEATURE_NAME_COL),
    sf.col(AGGREGATED_COL),
    sf.col(AGGREGATED_VALUE_COL),
    sf.col(VALUE_COL).alias(METRIC_VALUE_COL),
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
    FEATURE_NAME_COL: F.col(FEATURE_NAME_COL),
    WEEK_COL: F.lit(current_features_iso_year_week),
}
final_uni_drift_report = create_entity_map_column(
    final_uni_drift_report, entity_mapping, ENTITY_MAP_COL
)

# COMMAND ----------

# DBTITLE 1,Compose a deterministic record id per row
# Add record_id column
final_uni_drift_report = compose_record_id(
    final_uni_drift_report,
    [WEEK_COL, FEATURE_NAME_COL, AGGREGATED_COL, AGGREGATED_VALUE_COL, METRIC_NAME_COL],
    MODEL_URI_COL,
    RECORD_ID_COL,
)

# COMMAND ----------

# DBTITLE 1,Define final report to save
# final_uni_drift_report schema
# [record_id, model_uri, week, bkcc, metric_name, feature_name, metric_value, ldts, warning_threshold, critical_threshold, alert, group_col, calmonth, entity_map]

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
for group in groups:
    group_report = report.filter(sf.col(AGGREGATED_VALUE_COL) == group)
    delta_upsert(
        source_df=group_report,
        target_identifier=DRIFT_REPORT_TABLE,
        merge_keys=[RECORD_ID_COL],
        partition_col_or_cols=[
            MODEL_URI_COL,
            CALMONTH_COL,
        ],
        as_metastore=True,
    )
