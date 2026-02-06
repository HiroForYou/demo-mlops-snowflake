# %% [markdown]
# # Partitioned Inference Batch
# Load inference data, run TABLE(model!PREDICT(...) OVER (PARTITION BY stats_ntile_group)), save to INFERENCE_LOGS.

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.ml.registry import Registry
import time

session = get_active_session()
session.sql("USE DATABASE BD_AA_DEV").collect()
session.sql("USE SCHEMA SC_STORAGE_BMX_PS").collect()

registry = Registry(
    session=session,
    database_name="BD_AA_DEV",
    schema_name="SC_MODELS_BMX"
)

INFERENCE_SAMPLE_FRACTION = 0.01

print("✅ Connected to Snowflake")
print(f"   Database: {session.get_current_database()}")
print(f"   Schema: {session.get_current_schema()}")
if INFERENCE_SAMPLE_FRACTION:
    print(f"   ⚠️  Sampling: {INFERENCE_SAMPLE_FRACTION*100:.1f}% del dataset de inferencia")

# %% [markdown]
# ## 1. Verify model

# %%
print("\n" + "="*80)
print("🔍 VERIFYING PARTITIONED MODEL")
print("="*80)

model_ref = registry.get_model("UNI_BOX_REGRESSION_PARTITIONED")
model_version = model_ref.version("PRODUCTION")
print(f"✅ UNI_BOX_REGRESSION_PARTITIONED @ {model_version.version_name} (PRODUCTION)")

# %% [markdown]
# ## 2. Load inference data (directamente desde INFERENCE_DATASET_CLEANED)
# No se usa Feature Store: inferencia consume la tabla cleaned con las mismas columnas que validó 01 (compatibles con training).

# %%
print("\n" + "="*80)
print("📋 LOADING INFERENCE DATA (INFERENCE_DATASET_CLEANED)")
print("="*80)

inference_df = session.table("BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_DATASET_CLEANED")

partition_col = next((c for c in inference_df.columns if c.upper() == "STATS_NTILE_GROUP"), None)
if partition_col is None:
    raise ValueError("stats_ntile_group not found in inference dataset.")

if INFERENCE_SAMPLE_FRACTION and 0 < INFERENCE_SAMPLE_FRACTION < 1:
    inference_df = inference_df.sample(frac=INFERENCE_SAMPLE_FRACTION)

n_records = inference_df.count()
print(f"   Inference records: {n_records:,}")
inference_df.group_by(partition_col).count().sort(partition_col).show()

print(f"✅ Loaded {n_records:,} records from INFERENCE_DATASET_CLEANED (sin Feature Store)")
inference_df.select("customer_id", "week", "brand_pres_ret", partition_col, "sum_past_12_weeks", "week_of_year").show(5)

# %% [markdown]
# ## 3. Prepare inference input (misma exclusión que 02/04: solo features numéricas, sin metadata)

# %%
print("\n" + "="*80)
print("🔧 PREPARING INFERENCE INPUT")
print("="*80)

# Misma lista de exclusión que script 02 (Feature Store) y 04: no son features para el modelo
excluded_cols = [
    "customer_id",
    "brand_pres_ret",
    "week",
    "group",
    "stats_group",
    "percentile_group",
    "stats_ntile_group",
    "FEATURE_TIMESTAMP",
]
excluded_upper = {c.upper() for c in excluded_cols}

# Obtener esquema de la tabla de inferencia para identificar solo columnas de features (numéricas, no excluidas)
inference_schema = session.sql(
    "DESCRIBE TABLE BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_DATASET_CLEANED"
).collect()
col_type_dict = {row["name"].upper(): str(row["type"]).upper() for row in inference_schema}
all_cols = [row["name"] for row in inference_schema]

NUMERIC_PREFIXES = ("FLOAT", "NUMBER", "INTEGER", "BIGINT", "DOUBLE")
feature_cols_actual = [
    c for c in all_cols
    if c.upper() not in excluded_upper
    and (col_type_dict.get(c.upper()) or "").startswith(NUMERIC_PREFIXES)
]

# Crear INFERENCE_INPUT_TEMP solo con columnas necesarias: claves + partition + features (lo que espera PREDICT)
# Excluir group, stats_group, percentile_group y cualquier no-feature (VARCHAR como PROD_KEY) para que PREDICT reciba la firma correcta
keys_and_partition_upper = {"CUSTOMER_ID", "BRAND_PRES_RET", "WEEK", partition_col.upper()}
feature_names_upper = {c.upper() for c in feature_cols_actual}
# Solo incluir columnas que sean: (1) claves/partición, o (2) features numéricas (verificar tipo también)
cols_to_keep = []
for c in inference_df.columns:
    c_upper = c.upper()
    if c_upper in keys_and_partition_upper:
        cols_to_keep.append(c)
    elif c_upper in feature_names_upper:
        # Verificar que realmente sea numérica (doble verificación)
        col_type = col_type_dict.get(c_upper, "")
        if col_type.startswith(NUMERIC_PREFIXES):
            cols_to_keep.append(c)

inference_input_df = inference_df.select(cols_to_keep)
inference_input_df.write.mode("overwrite").save_as_table("BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_INPUT_TEMP")

temp_table = session.table("BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_INPUT_TEMP")
actual_cols = temp_table.columns
customer_id_col = next((c for c in actual_cols if c.upper() == "CUSTOMER_ID"), "CUSTOMER_ID")
brand_col = next((c for c in actual_cols if c.upper() == "BRAND_PRES_RET"), "BRAND_PRES_RET")
week_col = next((c for c in actual_cols if c.upper() == "WEEK"), "WEEK")
partition_col_actual = next((c for c in actual_cols if c.upper() == partition_col.upper()), partition_col)

# Recalcular feature_cols_actual desde la temp table (asegurar que solo tenga numéricas, sin VARCHAR como PROD_KEY)
temp_schema = session.sql("DESCRIBE TABLE BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_INPUT_TEMP").collect()
temp_col_type_dict = {row["name"].upper(): str(row["type"]).upper() for row in temp_schema}
feature_cols_actual = [
    c for c in actual_cols
    if c.upper() not in excluded_upper
    and (temp_col_type_dict.get(c.upper()) or "").startswith(NUMERIC_PREFIXES)
]

# Verificar que no haya VARCHARs en features (debug)
non_numeric_in_features = [
    c for c in feature_cols_actual
    if not (temp_col_type_dict.get(c.upper()) or "").startswith(NUMERIC_PREFIXES)
]
if non_numeric_in_features:
    print(f"⚠️  ADVERTENCIA: Columnas no numéricas en features: {non_numeric_in_features}")
    feature_cols_actual = [c for c in feature_cols_actual if c not in non_numeric_in_features]

print(f"✅ Excluidas (no features): {list(excluded_cols)}")
print(f"✅ {len(feature_cols_actual)} features numéricas para PREDICT, partition: {partition_col_actual}")
if len(feature_cols_actual) > 0:
    print(f"   Primeras 5 features: {feature_cols_actual[:5]}")

# %% [markdown]
# ## 4. Execute partitioned inference

# %%
print("\n" + "="*80)
print("🚀 EXECUTING PARTITIONED INFERENCE")
print("="*80)

start_time = time.time()
# Pass columns as-is; model was registered with sample_input from training (same types).
feature_list = ", ".join(f"i.{col}" for col in feature_cols_actual)

predictions_sql = f"""
WITH model_predictions AS (
    SELECT 
        p.customer_id,
        p.{partition_col_actual},
        p.predicted_uni_box_week
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_INPUT_TEMP i,
        TABLE(
            BD_AA_DEV.SC_MODELS_BMX.UNI_BOX_REGRESSION_PARTITIONED!PREDICT(
                i.{customer_id_col},
                i.{partition_col_actual},
                {feature_list}
            ) OVER (PARTITION BY i.{partition_col_actual})
        ) p
)
SELECT 
    mp.customer_id,
    mp.{partition_col_actual},
    i.{week_col},
    i.{brand_col},
    ROUND(mp.predicted_uni_box_week, 2) AS predicted_uni_box_week
FROM model_predictions mp
JOIN BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_INPUT_TEMP i 
    ON mp.customer_id = i.{customer_id_col}
    AND mp.{partition_col_actual} = i.{partition_col_actual}
ORDER BY mp.{partition_col_actual}, mp.customer_id
"""

predictions_df = session.sql(predictions_sql)
prediction_count = predictions_df.count()
inference_time = time.time() - start_time

print(f"✅ Done in {inference_time:.2f}s — {prediction_count:,} predictions")
predictions_df.show(10)

# %% [markdown]
# ## 5. Statistics

# %%
stats_sql = f"""
WITH model_predictions AS (
    SELECT 
        p.customer_id,
        p.predicted_uni_box_week
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_INPUT_TEMP i,
        TABLE(
            BD_AA_DEV.SC_MODELS_BMX.UNI_BOX_REGRESSION_PARTITIONED!PREDICT(
                i.{customer_id_col},
                i.{partition_col_actual},
                {feature_list}
            ) OVER (PARTITION BY i.{partition_col_actual})
        ) p
)
SELECT
    COUNT(*) AS TOTAL_PREDICTIONS,
    COUNT(DISTINCT customer_id) AS UNIQUE_CUSTOMERS,
    ROUND(MIN(predicted_uni_box_week), 2) AS MIN_PREDICTION,
    ROUND(MAX(predicted_uni_box_week), 2) AS MAX_PREDICTION,
    ROUND(AVG(predicted_uni_box_week), 2) AS AVG_PREDICTION,
    ROUND(STDDEV(predicted_uni_box_week), 2) AS STDDEV_PREDICTION,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY predicted_uni_box_week), 2) AS Q1,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY predicted_uni_box_week), 2) AS MEDIAN,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY predicted_uni_box_week), 2) AS Q3
FROM model_predictions
"""

session.sql(stats_sql).show()

# %% [markdown]
# ## 6. Save to INFERENCE_LOGS

# %%
session.sql("""
    CREATE TABLE IF NOT EXISTS BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_LOGS (
        customer_id VARCHAR,
        week VARCHAR,
        brand_pres_ret VARCHAR,
        stats_ntile_group VARCHAR,
        predicted_uni_box_week FLOAT,
        inference_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
        model_version VARCHAR
    )
""").collect()
insert_sql = f"""
INSERT INTO BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_LOGS
    (customer_id, week, brand_pres_ret, stats_ntile_group, predicted_uni_box_week, model_version)
WITH model_predictions AS (
    SELECT 
        p.customer_id,
        p.{partition_col_actual},
        p.predicted_uni_box_week
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_INPUT_TEMP i,
        TABLE(
            BD_AA_DEV.SC_MODELS_BMX.UNI_BOX_REGRESSION_PARTITIONED!PREDICT(
                i.{customer_id_col},
                i.{partition_col_actual},
                {feature_list}
            ) OVER (PARTITION BY i.{partition_col_actual})
        ) p
)
SELECT 
    mp.customer_id,
    i.{week_col},
    i.{brand_col},
    mp.{partition_col_actual},
    mp.predicted_uni_box_week,
    '{model_version.version_name}'
FROM model_predictions mp
JOIN BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_INPUT_TEMP i 
    ON mp.customer_id = i.{customer_id_col}
    AND mp.{partition_col_actual} = i.{partition_col_actual}
"""

session.sql(insert_sql).collect()
log_count = session.sql("SELECT COUNT(*) as CNT FROM BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_LOGS").collect()[0]['CNT']
print(f"✅ Saved {log_count:,} to INFERENCE_LOGS")
session.sql("SELECT * FROM BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_LOGS ORDER BY inference_timestamp DESC LIMIT 5").show()

# %% [markdown]
# ## 7. Summary

# %%
print(f"\n🎉 Done — {prediction_count:,} predictions, {inference_time:.2f}s, model {model_version.version_name}")
