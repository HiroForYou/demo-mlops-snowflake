# Databricks notebook source
# MAGIC %md # Forecast Train
# MAGIC Objective: This notebook trains and logs each sale forecast model.
# MAGIC
# MAGIC Input widgets:
# MAGIC - main_dir: Main execution environment, usually is "/mnt/{env}/PROCESSED/MODELS/"
# MAGIC - model_name: Identifier of the execution, ex: feature_eng_weather
# MAGIC - cutoff_date: Date that will be used as an identifier of the model name. Must be a sunday. This date will be used to filter the training data. Ex: 20221211
# MAGIC
# MAGIC Input: 
# MAGIC - Train set
# MAGIC
# MAGIC Output:
# MAGIC - A dataframe containing the records per group
# MAGIC - MLFlow experiment per group
# MAGIC - Experiment-Group map dataframe
# MAGIC - Serialized autoML experiments in a dbfs folder
# MAGIC
# MAGIC Process: 
# MAGIC 1. Save a separate train set per group
# MAGIC 2. Train a regression model per dataset
# MAGIC 3. Export the best run per experiment to a dbfs folder

# COMMAND ----------

# DBTITLE 1,Import packages
import os
import uuid
import pprint
from databricks import automl
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window as W
from pyspark.sql.types import BooleanType, IntegerType

# COMMAND ----------

# DBTITLE 1,Load widgets
# Main execution directory, usually is "/mnt/{env}/PROCESSED/MODELS/".
main_dir = dbutils.widgets.get("main_dir")
# Identifier of the execution, ex: feature_eng_weather.
model_name = dbutils.widgets.get("model_name")
# Filter threshold for max date, ex: 20221212.
cutoff_date = dbutils.widgets.get("cutoff_date")
# Databricks environment to use. Options "dev" or "prod".
environment = dbutils.widgets.get("environment")

print("main_dir: ", main_dir)
print("model_name: ", model_name)
print("cutoff_date: ", cutoff_date)
print("environment", environment)

if environment == "prod":
    raise ValueError("Training is currently only supported in dev")

# COMMAND ----------

# DBTITLE 1,Load MLflow utils
# MAGIC %run ../../../../advanced_analytics/mlflow

# COMMAND ----------

# DBTITLE 1,Notebook parameters
# Where everything will be saved
model_identifier = f"{model_name}_{cutoff_date}"
model_dir = os.path.join(main_dir, model_identifier)

# Factor to sample train set per group
sample_rate = 0.2

# COMMAND ----------

# DBTITLE 1,Input paths
# Single delta file containing the whole train set
train_set_path = os.path.join(model_dir, "train_dataset")

# COMMAND ----------

# DBTITLE 1,Output paths
group_train_set_path_directory = os.path.join(model_dir, "train_dataset_per_group")
group_submodel_path = os.path.join(model_dir, "group_submodel_map")
experiment_submodel_path = os.path.join(model_dir, "experiment_submodel_map")

# COMMAND ----------

# DBTITLE 1,Separating trainsets per line type
# This cell creates a column "group" that assigns each record to a certain
# model based on the previous sale pattern.
train_set = (
    spark.read.load(train_set_path)
    .withColumn(
        "group",
        F.concat(
            F.col("w_m1_total").cast(BooleanType()).cast(IntegerType()),
            F.col("w_m2_total").cast(BooleanType()).cast(IntegerType()),
            F.col("w_m3_total").cast(BooleanType()).cast(IntegerType()),
            F.col("w_m4_total").cast(BooleanType()).cast(IntegerType()),
        ),
    )
    .filter(F.col("group") != "0000")
    .withColumn(
        "sum_p4w",
        F.col("w_m1_total")
        + F.col("w_m2_total")
        + F.col("w_m3_total")
        + F.col("w_m4_total"),
    )
    .persist()
)

# COMMAND ----------

# DBTITLE 1,Group by percentiles-sales pattern
# In this command, percentiles are calculated within each of the
# 15 groups by weekly sales. Subsequently, the previous 15 groups
# are regrouped into 4 groups that capture similar sales patterns
# and percentiles.

group_stat_0 = (
    (F.col("q1_uni_box_week") == 0)
    & (F.col("q2_uni_box_week") == 0)
    & (F.col("q3_uni_box_week") == 0)
)
group_stat_1 = (
    (F.col("q1_uni_box_week") == 0)
    & (F.col("q2_uni_box_week") == 0)
    & (F.col("q3_uni_box_week") > 0)
)
group_stat_2 = (
    (F.col("q1_uni_box_week") == 0)
    & (F.col("q2_uni_box_week") > 0)
    & (F.col("q3_uni_box_week") > 0)
)
group_stat_3 = (
    (F.col("q1_uni_box_week") > 0)
    & (F.col("q2_uni_box_week") > 0)
    & (F.col("q3_uni_box_week") > 0)
)

stats = (
    train_set.groupby("group")
    .agg(
        F.percentile_approx("uni_box_week", 0.25, 1000).alias("q1_uni_box_week"),
        F.percentile_approx("uni_box_week", 0.5, 1000).alias("q2_uni_box_week"),
        F.percentile_approx("uni_box_week", 0.75, 1000).alias("q3_uni_box_week"),
    )
    .withColumn(
        "stats_group",
        F.when(group_stat_0, "group_stat_0")
        .when(group_stat_1, "group_stat_1")
        .when(group_stat_2, "group_stat_2")
        .when(group_stat_3, "group_stat_3")
        # You messed up if any percentile turned out to be negative
        .otherwise(None),
    )
    .select("group", "stats_group")
)
train_set = train_set.join(other=stats, on="group", how="inner")

# COMMAND ----------

# DBTITLE 1,Group by percentiles
# In this command, the 4 groups of the previous command are used
# as input and percentiles are calculated again, but using the
# sales of the last 4 weeks as a base.
# This generates 16 groups that are differentiated by sales magnitude,
# which seeks to group time series with similar sales patterns and magnitudes.

percentiles = 4
group_col = "stats_group"
value_col = "sum_p4w"

percentile_ranking = F.ntile(percentiles).over(
    W.partitionBy(group_col).orderBy(value_col)
)
train_set = train_set.withColumn("percentile_group", percentile_ranking).withColumn(
    "stats_ntile_group", F.concat_ws("_", "stats_group", "percentile_group")
)

# COMMAND ----------

# DBTITLE 1,Collect distinct groups
groups = (
    train_set.groupby("stats_ntile_group")
    .agg(F.first("customer_id"))
    .drop("customer_id")
    .toPandas()["stats_ntile_group"]
    .values
)

# COMMAND ----------

# DBTITLE 1,Save per group
# Save a sampled portion of each group into separate directories
# with a suffix. Each of them will be used in a separate autoML experiment.
for group in groups:
    group_set_dir = os.path.join(group_train_set_path_directory, f"train_set_{group}")
    try:
        dbutils.fs.ls(group_set_dir)
        print(f"Found group {group}. Skipping...")
        continue
    except Exception:
        pass
    print(f"Currently writing group {group}")
    train_set.filter(F.col("stats_ntile_group") == group).sample(
        sample_rate
    ).write.mode("overwrite").option("header", "true").option(
        "overwriteSchema", "true"
    ).save(
        group_set_dir
    )
    print("Done")

# COMMAND ----------

# This cell iterates on all groups and executes a databricks
# automl experiment and logs its information to a dictionary.

# Dictionary to save the best model_id per group (Group: model_id)
try:
    experiment_submodel_map = {
        row["group"]: row["experiment_dir"]
        for row in spark.read.load(experiment_submodel_path)
        .select("group", "experiment_dir")
        .collect()
    }
    group_submodel_map = {
        row["group"]: row["model"]
        for row in spark.read.load(group_submodel_path)
        .select("group", "model")
        .collect()
    }
except:
    experiment_submodel_map = {}
    group_submodel_map = {}

user_id = spark.sql("SELECT current_user() AS user").collect()[0]["user"]
base_experiment_dir = f"/Users/{user_id}"

for group in groups:
    print("\n" + "*" * 60)
    print(f"Processing group: {group}")
    print("*" * 60)

    if group in group_submodel_map:
        print(
            f"Group {group} already present in the table (model: {group_submodel_map[group]}). Skipping training."
        )
        continue

    automl_experiment = f"{model_identifier}_{group}_uni_box_{uuid.uuid4().hex}"
    automl_experiment_dir = os.path.join(base_experiment_dir, automl_experiment)

    group_dataset_path = os.path.join(
        group_train_set_path_directory, f"train_set_{group}"
    )
    train_set = spark.read.load(group_dataset_path)

    try:
        # Train and get the best model
        summary = automl.regress(
            # AutoML produces a valid path given a dir and a name
            experiment_dir=base_experiment_dir,
            experiment_name=automl_experiment,
            dataset=train_set,
            target_col="uni_box_week",
            exclude_cols=[
                "customer_id",
                "brand_pres_ret",
                "week",
                "group",
                "stats_group",
                "percentile_group",
                "stats_ntile_group",
            ],
            primary_metric="rmse",
            timeout_minutes=60,
        )
    except ValueError as e:
        print(f"Training failed for {group}: {e}.")

    # Always retrieve the model with the smallest generalization
    # loss
    best_run_uri = get_best_possible_uri(
        full_experiment_dir=automl_experiment_dir,
        metric_name="test_root_mean_squared_error",
        is_loss=True,
    )
    print(f"AutoML completed successfully for group {group}. Best run: {best_run_uri}")
    # Keep track of the model_id of current group
    experiment_submodel_map[group] = automl_experiment_dir
    group_submodel_map[group] = best_run_uri
    try:
        rows_group = [T.Row(group=k, model=v) for k, v in group_submodel_map.items()]
        schema_group = T.StructType(
            [
                T.StructField("group", T.StringType(), nullable=False),
                T.StructField("model", T.StringType(), nullable=False),
            ]
        )
        rows_experiment = [
            T.Row(group=k, model=v) for k, v in experiment_submodel_map.items()
        ]
        schema_experiment = T.StructType(
            [
                T.StructField("group", T.StringType(), nullable=False),
                T.StructField("experiment_dir", T.StringType(), nullable=False),
            ]
        )
        spark.createDataFrame(rows_group, schema=schema_group).write.mode(
            "overwrite"
        ).save(group_submodel_path)
        spark.createDataFrame(rows_experiment, schema=schema_experiment).write.mode(
            "overwrite"
        ).save(experiment_submodel_path)
        print(f"Save successful. Entry added: {group} -> {best_run_uri}")
    except Exception as e:
        print(f"Error saving the map for group {group}: {e}")
    pprint.pprint(group_submodel_map)
