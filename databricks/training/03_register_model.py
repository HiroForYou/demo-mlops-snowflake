# Databricks notebook source
# MAGIC %md
# MAGIC # Forecast Register
# MAGIC
# MAGIC Objective: This notebook registers the new trained ensemble
# MAGIC
# MAGIC Input widgets:
# MAGIC - main_dir (str): main execution environment, usually is "/mnt/{env}/PROCESSED/MODELS/"
# MAGIC - model_name (str): name identifier of the execution, ex: feature_eng_weather
# MAGIC - cutoff_date: Date that will be used as an identifier of the model name. Must be a sunday. This date will be used to filter the training data. Ex: 20221211
# MAGIC - environment (str): environment in which the script is run, either dev or prod
# MAGIC
# MAGIC Input: 
# MAGIC - Training Datasets per Group
# MAGIC - Group to Submodel Map created during training
# MAGIC
# MAGIC Output:
# MAGIC - Registered model in UC
# MAGIC
# MAGIC Process: 
# MAGIC 1. Read group training datasets and recreate full training dataset
# MAGIC 2. Read the group submodel map
# MAGIC 3. Create an ensemble model and register it in Unity Catalog
# MAGIC

# COMMAND ----------

# DBTITLE 1,Imports
from typing import Dict
import datetime
from functools import reduce
import sys
import os
import shutil

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import PythonModelContext
import pandas as pd

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# DBTITLE 1,Widgets
main_dir = dbutils.widgets.get("main_dir")
model_name = dbutils.widgets.get("model_name")
cutoff_date = dbutils.widgets.get("cutoff_date")
environment = dbutils.widgets.get("environment")
model_execution_date = dbutils.widgets.get("execution_date")

print("main_dir: ", main_dir)
print("model_name: ", model_name)
print("cutoff_date: ", cutoff_date)
print("model_execution_date: ", model_execution_date)

# COMMAND ----------

# DBTITLE 1,Import Workaround
if environment == "dev":
    library_prefix = "/Workspace/Repos/juan.echeagaray@arcacontal.com"
elif environment == "prod":
    raise ValueError("Training only occurs in dev")

LIBRARY_PATH = os.path.join(
    library_prefix, "Analytics-DS-ML-Toolkit/advanced_analytics"
)
MLFLOW_LIBRARY_PATH = os.path.join(LIBRARY_PATH, "advanced_analytics")

if LIBRARY_PATH not in sys.path:
    sys.path.append(LIBRARY_PATH)

from advanced_analytics.ensemble import PandasEnsemble
from advanced_analytics.mlflow import (
    set_tags_and_alias,
    get_latest_registered_version,
    create_experiment,
)

# COMMAND ----------

# DBTITLE 1,Parameters
model_dir = os.path.join(main_dir, f"{model_name}_{cutoff_date}")
group_train_set_dir = os.path.join(model_dir, "train_dataset_per_group")
group_submodel_path = os.path.join(model_dir, "group_submodel_map")
temp_submodel_artifact_path = os.path.join(model_dir, "temp_models")

user_id = spark.sql("SELECT current_user() AS user_id").collect()[0]["user_id"]
main_experiment_name = os.path.join("/Users", user_id, model_name)
registered_model_name = f"models_dev.mx.{model_name}"
try:
    model_execution_date_fmt = datetime.datetime.strptime(
        model_execution_date, "%Y-%m-%d"
    ).strftime("%Y%m%d")
except ValueError:
    model_execution_date_fmt = model_execution_date

# COMMAND ----------

# DBTITLE 1,Create experiment to hold the Main model
main_experiment_id = create_experiment(experiment_name=main_experiment_name)

# COMMAND ----------

# DBTITLE 1,Recreate training set
group_train_paths = [
    f.path.replace("dbfs:", "") for f in dbutils.fs.ls(group_train_set_dir)
]
train_set_list = []
for path in group_train_paths:
    train_set_list.append(spark.read.load(path))

train_set = reduce(lambda x, y: x.unionByName(y), train_set_list).drop(
    *[
        "customer_id",
        "week",
        "brand_pres_ret",
        "stats_group",
        "percentile_group",
        "group",
    ]
)

# COMMAND ----------

# DBTITLE 1,Read group submodel map
group_submodel_map = (
    spark.read.load(group_submodel_path)
    .toPandas()
    .set_index("group")["model"]
    .to_dict()
)

# COMMAND ----------

# DBTITLE 1,Instance to group mapping
def instance_to_group(df: pd.DataFrame) -> pd.DataFrame:
    df["group_id"] = df["stats_ntile_group"]
    return df

# COMMAND ----------

# DBTITLE 1,Download MLFlow artifacts to local dir
submodel_artifacts = {
    model_id: mlflow.artifacts.download_artifacts(model_uri)
    for model_id, model_uri in group_submodel_map.items()
}

# COMMAND ----------

# DBTITLE 1,Prepare Artifacts
mlflow.set_experiment(main_experiment_name)

input_example = train_set.limit(10)

dbutils.fs.mkdirs(temp_submodel_artifact_path)

model = PandasEnsemble(
    group_submodel_map=group_submodel_map,
    model_input_columns=train_set.drop("uni_box_week").columns,
    instance_to_group=instance_to_group,
)
artifacts = {}
for group_id, submodel_path in submodel_artifacts.items():
    # Specify dbfs as the file system, (cannot use os join since the temp
    # dir is already an "absolute" path)
    model_artifact_path = os.path.join(f"/dbfs{temp_submodel_artifact_path}", group_id)
    shutil.copytree(src=submodel_path, dst=model_artifact_path, dirs_exist_ok=True)
    artifacts[group_id] = model_artifact_path

all_dependencies = set()
for trained_submodel_path in artifacts.values():
    requirements_path = os.path.join(trained_submodel_path, "requirements.txt")
    with open(requirements_path, "r") as f:
        all_dependencies.update(f.read().splitlines())
extra_pip_requirements = list(all_dependencies)

model.load_context(PythonModelContext(artifacts))

group_submodel_estimator_cls = {
    group: pipe[-1].__class__.__name__ for group, pipe in model.models.items()
}

train_run_metadata = (
    {
        "train_set_path": group_train_set_dir,
        "group_submodel_path": group_submodel_path,
    }
    | {"group_submodel_classes": group_submodel_estimator_cls}
    | dbutils.widgets.getAll()
)

# COMMAND ----------

# DBTITLE 1,Model Registering
pd_sample = input_example.toPandas()
X = pd_sample.drop("uni_box_week", axis=1)
y_pred = model.predict(context=None, model_input=X)
signature = infer_signature(X, y_pred)
input_example = X.head(8)

with mlflow.start_run(experiment_id=main_experiment_id):

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model,
        signature=signature,
        artifacts=artifacts,
        input_example=input_example,
        code_path=[MLFLOW_LIBRARY_PATH],
        registered_model_name=registered_model_name,
        metadata=train_run_metadata,
        extra_pip_requirements=extra_pip_requirements,
    )
    mlflow.log_params(model.group_submodel_map)

dbutils.fs.rm(temp_submodel_artifact_path, recurse=True)

# COMMAND ----------

# DBTITLE 1,Set tags and alias for the registered model
model_tags = {
    "execution_date": model_execution_date_fmt,
    "cutoff_date": cutoff_date,
}
model_version = get_latest_registered_version(registered_model_name)
set_tags_and_alias(
    registered_model_name=registered_model_name,
    model_version=model_version,
    model_alias=model_execution_date_fmt,
    model_tags=model_tags,
)
