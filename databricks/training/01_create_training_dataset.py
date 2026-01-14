# Databricks notebook source
# MAGIC %md # Forecast Dataset
# MAGIC Objective: This notebook generates the train set for the sale forecast autoML process.
# MAGIC
# MAGIC Input widgets:
# MAGIC - main_dir: Main execution environment, usually is "/mnt/{env}/PROCESSED/MODELS/"
# MAGIC - model_name: Identifier of the execution, ex: feature_eng_weather
# MAGIC - cutoff_date: Date that will be used as an identifier of the model name. Must be a sunday. This date will be used to filter the training data. Ex: 20221211
# MAGIC - environment: Databricks environment to use. Options: dev or prod.
# MAGIC
# MAGIC Input: 
# MAGIC - Sales
# MAGIC - Customers
# MAGIC - Product Type Map
# MAGIC - Weather
# MAGIC - CEDI Structure
# MAGIC - Coolers
# MAGIC - Calendar
# MAGIC - SKU Code Change History Path
# MAGIC
# MAGIC Output:
# MAGIC - Delta file with train set records at a Customer-BPR-week level.
# MAGIC
# MAGIC Process: 
# MAGIC 1. Get raw sales data
# MAGIC 2. Clean it using the intelligent rounding function
# MAGIC 3. Create sale features, including lags and aggregations
# MAGIC 4. Add ISSCOM Channel feature
# MAGIC 5. Add coolers features
# MAGIC 6. Add Seasonality feature
# MAGIC 7. Add weather feature
# MAGIC 5. Save train dataset

# COMMAND ----------

# DBTITLE 1,Import packages
# MAGIC %scala
# MAGIC import org.apache.spark.sql.DataFrame
# MAGIC import org.apache.spark.sql.types._
# MAGIC import org.apache.spark.sql.functions._
# MAGIC
# MAGIC import java.nio.file.Paths

# COMMAND ----------

# DBTITLE 1,Loading path utilities
# MAGIC %run ../../../../scala_utils/ps_paths

# COMMAND ----------

# DBTITLE 1,Load widgets
# MAGIC %scala
# MAGIC // Main execution directory, usually is "/mnt/{env}/PROCESSED/MODELS/".
# MAGIC val main_dir = dbutils.widgets.get("main_dir")
# MAGIC // Identifier of the execution, ex: feature_eng_weather.
# MAGIC val model_name = dbutils.widgets.get("model_name")
# MAGIC // Execution date that will be used as an identifier of the model name.
# MAGIC // Must be a sunday. This date will be used as the execution date so the
# MAGIC // appropriate backup data is retrieved. Ex: 20221211.
# MAGIC val cutoff_date = dbutils.widgets.get("cutoff_date")
# MAGIC // Databricks environment to use. Options: dev or prod.
# MAGIC val environment = dbutils.widgets.get("environment")

# COMMAND ----------

# DBTITLE 1,Notebook parameters
# MAGIC %scala
# MAGIC // Most recent week
# MAGIC val sunday = cutoff_date.replaceAll("-", "")
# MAGIC val model_dir = Paths.get(main_dir, s"${model_name}_$cutoff_date").toString()
# MAGIC
# MAGIC // Change prefix depending on environment
# MAGIC val current_dl_prefix = "/mnt/DL"
# MAGIC val dl_env = if (environment != "dev") current_dl_prefix else "/mnt/DLP"
# MAGIC val dl_input_prefix = Paths.get(dl_env, "PROCESSED/INPUT").toString()
# MAGIC
# MAGIC val dates_path = DATES_PATH

# COMMAND ----------

# DBTITLE 1,Paths DF
# MAGIC %scala
# MAGIC // Define input paths that will be used in the execution
# MAGIC val pathsDF = Seq(
# MAGIC   ("sales_path", "prd.gold.v_sales_mx", "unity_catalog"),
# MAGIC   ("products_path", "prd.gold.v_products_mx", "unity_catalog"),
# MAGIC   ("customers_path", "prd.gold.v_customers_mx", "unity_catalog"),
# MAGIC   ("customers_cedis_path", "prd.gold.v_customers_cedis_mx", "unity_catalog"),
# MAGIC   ("dates_path", dates_path, "delta"),
# MAGIC   (
# MAGIC     "weather_history_path",
# MAGIC     Paths.get(dl_input_prefix, "WEATHER/DAILY/delta/").toString(),
# MAGIC     "delta"
# MAGIC   ),
# MAGIC   (
# MAGIC     "coolers_path",
# MAGIC     Paths.get(dl_input_prefix, "COOLERS/delta/").toString(),
# MAGIC     "delta"
# MAGIC   ),
# MAGIC   (
# MAGIC     "sku_code_change_history_path",
# MAGIC     Paths.get(dl_input_prefix, "CAMBIOS_SKU/delta").toString(),
# MAGIC     "delta"
# MAGIC   ),
# MAGIC   (
# MAGIC     "cedi_structure_path",
# MAGIC     getHistorical(
# MAGIC       main = Paths.get(dl_env, "/PROCESSED/INPUT/CEDISTRUCTURE/").toString(),
# MAGIC       prod_file_name = "delta",
# MAGIC       sunday = sunday,
# MAGIC       prefix_backup_file_name = ""
# MAGIC     ),
# MAGIC     "delta"
# MAGIC   )
# MAGIC )
# MAGIC   .toDF("var", "path", "type")

# COMMAND ----------

# DBTITLE 1,Input Paths
# MAGIC %scala
# MAGIC val products_path = getPath(pathsDF, "products_path")
# MAGIC val sales_path = getPath(pathsDF, "sales_path")
# MAGIC val customers_path = getPath(pathsDF, "customers_path")
# MAGIC val customers_cedis_path = getPath(pathsDF, "customers_cedis_path")
# MAGIC val sku_code_change_history_path = getPath(pathsDF, "sku_code_change_history_path")
# MAGIC val weather_history_path = getPath(pathsDF, "weather_history_path")
# MAGIC val cedi_structure_path = getPath(pathsDF, "cedi_structure_path")
# MAGIC val coolers_path = getPath(pathsDF, "coolers_path")

# COMMAND ----------

# DBTITLE 1,Output paths
# MAGIC %scala
# MAGIC // Where the train dataset will be saved
# MAGIC var train_set_save_path =
# MAGIC   Paths.get(model_dir, "train_dataset").toString()
# MAGIC // Temporal, erased at the end
# MAGIC var last_weeks_filtered_sales =
# MAGIC   Paths.get(model_dir, "aux_train_dataset").toString()

# COMMAND ----------

# MAGIC // HERE GOES THE DATASET CREATION

# COMMAND ----------

# MAGIC %md ### Save trainset

# COMMAND ----------

# DBTITLE 1,Save train set
# MAGIC %scala
# MAGIC sales.write
# MAGIC   .format("delta")
# MAGIC   .mode("overwrite")
# MAGIC   .option("header", "true")
# MAGIC   .option("overwriteSchema", "true")
# MAGIC   .save(train_set_save_path)

# COMMAND ----------

# MAGIC %scala
# MAGIC dbutils.fs.rm(last_weeks_filtered_sales, true)
