// Databricks notebook source
// MAGIC %md # Forecast Inference Dataset
// MAGIC
// MAGIC Objective: This notebook generates the test set to be predicted
// MAGIC
// MAGIC Input widgets:
// MAGIC - main_dir: Main directory in which project is saved. Ex: /mnt/DL/PROCESSED/MODELS_UC/FORECAST_BPR_CUSTOMER_WEEK/
// MAGIC - execution_date: Execution date of algorithm, must be a sunday in YYYYMMDD format. Ex: 20220206 corresponds to February 6th, 2022. (A sunday).
// MAGIC
// MAGIC Input:
// MAGIC - Sales
// MAGIC - Customers
// MAGIC - Products
// MAGIC - Weather
// MAGIC - CEDI Structure
// MAGIC - Coolers
// MAGIC - Calendar
// MAGIC - SKU Code Change History Path
// MAGIC
// MAGIC Output:
// MAGIC - Delta file with inference set records at a Customer-BPR-week level.
// MAGIC
// MAGIC Process:
// MAGIC 1. Get raw sales data
// MAGIC 2. Clean it using the intelligent rounding function
// MAGIC 3. Create sale features, including lags and aggregations
// MAGIC 4. Add ISSCOM Channel feature
// MAGIC 5. Add coolers features
// MAGIC 6. Add Seasonality feature
// MAGIC 7. Add weather feature
// MAGIC 8. Save inference dataset

// COMMAND ----------

// DBTITLE 1,Import packages
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import java.nio.file.Paths
import java.io.FileNotFoundException
import io.delta.tables.DeltaTable

// COMMAND ----------

// DBTITLE 1,Load calendar utilities
// MAGIC %run ../../../../scala_utils/ps_calendar

// COMMAND ----------

// DBTITLE 1,Load widgets
// Set week directory path
// Main directory
val main_dir = dbutils.widgets.get("main_dir")
// Execution execution_date
val execution_date = dbutils.widgets.get("execution_date")

// COMMAND ----------

// DBTITLE 1,Notebook parameters
var predicted_week = get_predicted_week(execution_date)
val dl_env = if (dbutils.secrets.get("KV", "ENV") == "PRD") "DL" else "DLP"
val dl_input_prefix = s"/mnt/$dl_env/PROCESSED/INPUT"

spark.conf.set("spark.sql.session.timeZone", "America/Mexico_City")

// COMMAND ----------

// DBTITLE 1,Input paths
// To run sales utils
val dates_path = DATES_PATH
val products_path = "prd.gold.v_products_mx"
val sales_path = "prd.gold.v_sales_mx"
val customers_path = "prd.gold.v_customers_mx"
val customers_cedis_path = "prd.gold.v_customers_cedis_mx"
val sku_code_change_history_path =
  Paths.get(dl_input_prefix, "CAMBIOS_SKU/delta").toString()
// Weather:
val weather_forecast_path =
  Paths.get(dl_input_prefix, "WEATHER/FORECAST/delta").toString()
val cedi_structure_path = "prd.gold.v_cedi_mx"
// Coolers
val coolers_path = Paths.get(dl_input_prefix, "COOLERS/delta").toString()

// COMMAND ----------

// DBTITLE 1,Set up Output Directory
try {
  val _ = dbutils.fs.ls(main_dir)
} catch {
  case e: FileNotFoundException => dbutils.fs.mkdirs(main_dir)
}

// COMMAND ----------

// DBTITLE 1,Output paths
var forecast_inference_dataset_path =
  Paths.get(main_dir, "FEATURES").toString()

// COMMAND ----------

# MAGIC // HERE GOES THE DATASET CREATION

// COMMAND ----------

// MAGIC %md
// MAGIC ### Add groups

// COMMAND ----------

var features_with_groups = featurized_predicted_week.withColumn(
    "group",
    concat(
      $"w_m1_total".cast(BooleanType).cast(IntegerType),
      $"w_m2_total".cast(BooleanType).cast(IntegerType),
      $"w_m3_total".cast(BooleanType).cast(IntegerType),
      $"w_m4_total".cast(BooleanType).cast(IntegerType)
    )
  )
  .filter(
    not($"group".isin("0000"))
  ) // We don't predict those with 0 sales in p4w
  .na
  .drop
  .withColumn(
    "stats_group",
    when($"group".isin("0001", "0010", "0100", "1000"), "group_stat_0")
      .otherwise(
        when(
          $"group".isin("0011", "0101", "0110", "1001", "1010", "1100"),
          "group_stat_1"
        ).otherwise(
          when($"group".isin("0111", "1011", "1101", "1110"), "group_stat_2")
            .otherwise(when($"group".isin("1111"), "group_stat_3"))
        )
      )
  )
  .withColumn(
    "sum_p4w",
    $"w_m1_total" + $"w_m2_total" + $"w_m3_total" + $"w_m4_total"
  )

val percentiles = 4
val groupCol = "stats_group"
val valueCol = "sum_past_4_weeks"

val percentileRanking = ntile(percentiles).over(
  Window.partitionBy(col(groupCol)).orderBy(col(valueCol))
)

features_with_groups = features_with_groups
  .withColumn("percentile_group", percentileRanking)
  .withColumn(
    "stats_ntile_group",
    concat($"stats_group", lit("_"), $"percentile_group")
  )
  .withColumn("timestamp", current_timestamp().alias("timestamp"))
  .drop("uni_box_week")

// COMMAND ----------

// MAGIC %md ### Save dataset

// COMMAND ----------

// Ensure the target dirctory exists
dbutils.fs.mkdirs(forecast_inference_dataset_path)

// If this is the first run, create a new Delta table
if (!DeltaTable.isDeltaTable(spark, forecast_inference_dataset_path)) {
  features_with_groups
    .write
    .format("delta")
    .mode("overwrite")
    .partitionBy("week")
    .save(forecast_inference_dataset_path)
} else {
  // If this is the first run, create a new Delta table for forecasts
  val deltaTable = DeltaTable.forPath(spark, forecast_inference_dataset_path)
  
  deltaTable
    .as("target")
    .merge(
      features_with_groups.as("source"),
      """
        target.customer_id = source.customer_id AND
        target.brand_pres_ret = source.brand_pres_ret AND
        target.week           = source.week
      """
    )
    .whenMatched()
      .updateAll()
    .whenNotMatched()
      .insertAll()
    .execute()
}
