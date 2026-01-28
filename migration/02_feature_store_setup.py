# %% [markdown]
# # Migration: Feature Store Setup (manteniendo Feature Store, sin FeatureView / sin Dynamic Tables)
#
# ## Overview
# Este script **mantiene Feature Store** (schema + `FeatureStore` + `Entity`) pero evita `FeatureView`
# (que internamente puede crear Dynamic Tables) y en su lugar construye y materializa un dataset de
# features como **tabla normal** en Snowflake.
#
# ## What We'll Do:
# 1. Crear/asegurar schema destino para Feature Store
# 2. Inicializar `FeatureStore`
# 3. Registrar `Entity` (opcional)
# 4. Construir dataset de features desde `TRAIN_DATASET_CLEANED`
# 5. Materializar features en una tabla (CTAS / overwrite)

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.ml.feature_store import FeatureStore, Entity, CreationMode

session = get_active_session()

# Set context
session.sql("USE DATABASE BD_AA_DEV").collect()
session.sql("USE SCHEMA SC_STORAGE_BMX_PS").collect()

print(f"✅ Connected to Snowflake")
print(f"   Database: {session.get_current_database()}")
print(f"   Schema: {session.get_current_schema()}")

# %% [markdown]
# ## 1. Create Feature Store Schema

# %%
print("\n" + "=" * 80)
print("🏪 PREPARANDO ESQUEMA DE FEATURES (SIN FEATURE VIEW)")
print("=" * 80)

# Crear schema destino (si no tienes permiso para crear schema, fallará aquí)
session.sql("CREATE SCHEMA IF NOT EXISTS BD_AA_DEV.SC_FEATURES_BMX").collect()
session.sql("USE SCHEMA BD_AA_DEV.SC_FEATURES_BMX").collect()
print("\n✅ Schema listo: BD_AA_DEV.SC_FEATURES_BMX")

# Mantener Feature Store (sin FeatureView)
fs = FeatureStore(
    session=session,
    database="BD_AA_DEV",
    name="SC_FEATURES_BMX",
    default_warehouse="WH_AA_DEV_DS_SQL",
    creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
)
print("✅ Feature Store inicializado (sin FeatureView)")

# %% [markdown]
# ## 2. Define Entity (Optional)

# %%
print("\n" + "=" * 80)
print("👤 DEFINING ENTITIES")
print("=" * 80)

# Definir Entity (no crea Dynamic Tables; es metadata del Feature Store)
customer_product_entity = Entity(
    name="CUSTOMER_PRODUCT",
    join_keys=["customer_id", "brand_pres_ret"],
    desc="Customer-Product combination entity for uni_box_week regression",
)

try:
    fs.register_entity(customer_product_entity)
    print("✅ Entity 'CUSTOMER_PRODUCT' registrada")
except Exception as e:
    print(f"⚠️  Entity puede ya existir o no ser registrable: {str(e)[:120]}")

# %% [markdown]
# ## 3. Crear dataset de features desde tablas limpias

# %%
print("\n" + "=" * 80)
print("📋 CONSTRUYENDO DATASET DE FEATURES")
print("=" * 80)

# Define excluded columns (not features)
excluded_cols = [
    "customer_id",
    "brand_pres_ret",
    "week",
    "group",
    "stats_group",
    "percentile_group",
    "stats_ntile_group",
    "uni_box_week",  # Target variable - not a feature
]

# Get column names efficiently using DESCRIBE TABLE (more efficient than loading full table)
print("\n⏳ Getting column names from table schema...")
columns_info = session.sql(
    """
    DESCRIBE TABLE BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED
"""
).collect()

all_columns = [row["name"] for row in columns_info]

# Get feature columns (all columns except excluded and target)
excluded_cols_upper = [ex.upper() for ex in excluded_cols]
feature_columns = [
    col for col in all_columns 
    if col.upper() not in excluded_cols_upper
]

print(f"\n📋 Column Analysis:")
print(f"   Total columns: {len(all_columns)}")
print(f"   Excluded columns: {len(excluded_cols)}")
print(f"   Feature columns: {len(feature_columns)}")

print(f"\n📋 Excluded columns (not features):")
for col in excluded_cols:
    print(f"   - {col}")

# Crear query de features
# Selecciona dinámicamente todas las columnas de features (excluyendo metadata y target)
feature_cols_str = ",\n        ".join(feature_columns)

feature_df = session.sql(
    f"""
    SELECT
        customer_id,
        brand_pres_ret,
        week,
        {feature_cols_str},
        -- Timestamp for Feature Store
        CASE 
            WHEN week IS NOT NULL THEN 
                TRY_TO_TIMESTAMP_NTZ(week, 'YYYYWW')
            ELSE CURRENT_TIMESTAMP()
        END AS FEATURE_TIMESTAMP
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED
    WHERE customer_id IS NOT NULL
        AND brand_pres_ret IS NOT NULL
"""
)

print("✅ Feature query created")

# Count features
feature_count = feature_df.count()
print(f"   Total feature records: {feature_count:,}")

# %% [markdown]
# ## 4. Materializar Features en Tabla (sin FeatureView)

# %%
print("\n" + "=" * 80)
print("🧱 MATERIALIZANDO FEATURES EN TABLA (CTAS)")
print("=" * 80)

FEATURES_TABLE = "BD_AA_DEV.SC_FEATURES_BMX.UNI_BOX_FEATURES"
print(f"\n📝 Creando/Reemplazando tabla: {FEATURES_TABLE}")
feature_df.write.mode("overwrite").save_as_table(FEATURES_TABLE)
print("✅ Tabla de features creada (sin Dynamic Tables / sin Feature Views)")

print("\n📊 Muestra de features (5 filas):")
session.table(FEATURES_TABLE).limit(5).show()

# %% [markdown]
# ## 6. Summary

# %%
print("\n" + "=" * 80)
print("✅ FEATURE DATASET SETUP COMPLETE!")
print("=" * 80)

print("\n📋 Summary:")
print("   ✅ Feature schema: BD_AA_DEV.SC_FEATURES_BMX")
print(f"   ✅ Features table: {FEATURES_TABLE}")
# Count actual feature columns (excluding metadata columns)
metadata_cols = ['customer_id', 'brand_pres_ret', 'week', 'FEATURE_TIMESTAMP']
actual_feature_count = len([col for col in feature_df.columns if col.upper() not in [m.upper() for m in metadata_cols]])
print(f"   ✅ Features: {actual_feature_count} features")
print(f"   ✅ Excluded from features: {', '.join(excluded_cols)}")
print(f"   ✅ Total records: {feature_count:,}")

print("\n💡 Next Steps:")
print("   1. Run 03_hyperparameter_search.py (seguirá funcionando sin FeatureView)")
print("   2. Run 04_many_model_training.py (lo ajustaremos para no depender de FeatureView)")

print("\n" + "=" * 80)
