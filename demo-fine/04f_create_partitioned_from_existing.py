# %% [markdown]
# # ARCA Demo: Crear Modelo Particionado desde Modelos Existentes
# 
# ## Objetivo
# Reutilizar los 6 modelos ya entrenados y registrados para crear UN modelo particionado.
# 
# ## Modelos existentes:
# - WEEKLY_SALES_FORECAST_SEGMENT_1 (PRODUCTION)
# - WEEKLY_SALES_FORECAST_SEGMENT_2 (PRODUCTION)
# - WEEKLY_SALES_FORECAST_SEGMENT_3 (PRODUCTION)
# - WEEKLY_SALES_FORECAST_SEGMENT_4 (PRODUCTION)
# - WEEKLY_SALES_FORECAST_SEGMENT_5 (PRODUCTION)
# - WEEKLY_SALES_FORECAST_SEGMENT_6 (PRODUCTION)
# 
# ## Resultado:
# - UN modelo particionado que enruta automáticamente al sub-modelo correcto

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.ml.registry import Registry
from snowflake.ml.model import custom_model, task
import pandas as pd
import numpy as np
from datetime import datetime

session = get_active_session()

session.sql("USE WAREHOUSE ARCA_DEMO_WH").collect()
session.sql("USE DATABASE ARCA_BEVERAGE_DEMO").collect()
session.sql("USE SCHEMA ML_DATA").collect()

registry = Registry(
    session=session,
    database_name="ARCA_BEVERAGE_DEMO",
    schema_name="MODEL_REGISTRY"
)

print("✅ Conectado a Snowflake")
print(f"   Database: {session.get_current_database()}")
print(f"   Schema: {session.get_current_schema()}")

# %% [markdown]
# ## 1. Verificar Modelos Existentes

# %%
print("\n🔍 Verificando modelos existentes...\n")

segments = ['SEGMENT_1', 'SEGMENT_2', 'SEGMENT_3', 'SEGMENT_4', 'SEGMENT_5', 'SEGMENT_6']
model_info = {}

for seg in segments:
    model_name = f"WEEKLY_SALES_FORECAST_{seg}"
    try:
        model = registry.get_model(model_name)
        version = model.version("PRODUCTION")
        model_info[seg] = {
            'name': model_name,
            'version': version,
            'exists': True
        }
        print(f"✅ {model_name} - PRODUCTION version found")
    except Exception as e:
        model_info[seg] = {'exists': False, 'error': str(e)}
        print(f"❌ {model_name} - {str(e)[:50]}")

available_models = sum(1 for m in model_info.values() if m.get('exists'))
print(f"\n📊 Modelos disponibles: {available_models}/6")

# %% [markdown]
# ## 2. Cargar los Modelos desde el Registry

# %%
print("\n📦 Cargando modelos desde el Registry...\n")

loaded_models = {}

for seg in segments:
    if model_info[seg].get('exists'):
        try:
            model_name = f"WEEKLY_SALES_FORECAST_{seg}"
            mv = registry.get_model(model_name).version("PRODUCTION")
            
            # Load the actual model object
            native_model = mv.load()
            loaded_models[seg] = native_model
            
            print(f"✅ {seg}: {type(native_model).__name__} cargado")
        except Exception as e:
            print(f"❌ {seg}: Error al cargar - {str(e)[:60]}")

print(f"\n📊 Modelos cargados: {len(loaded_models)}/6")

# %% [markdown]
# ## 3. Definir Modelo Particionado

# %%
class PartitionedWeeklySalesForecast(custom_model.CustomModel):
    """
    Modelo particionado que enruta predicciones al sub-modelo correcto por segmento.
    """
    def __init__(self, model_context):
        super().__init__(model_context)
        self.feature_cols = [
            'CUSTOMER_TOTAL_UNITS_4W', 'WEEKS_WITH_PURCHASE', 'VOLUME_QUARTILE',
            'WEEK_OF_YEAR', 'MONTH', 'QUARTER', 'TRANSACTION_COUNT',
            'UNIQUE_PRODUCTS_PURCHASED', 'AVG_UNITS_PER_TRANSACTION'
        ]
    
    @custom_model.partitioned_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        if len(input_df) == 0:
            return pd.DataFrame(columns=['CUSTOMER_ID', 'SEGMENT', 'PREDICTED_WEEKLY_SALES'])
        
        segment = input_df['SEGMENT'].iloc[0]
        segment_key = f"segment_{segment.lower()}"
        
        segment_model = self.context.model_ref(segment_key)
        X = input_df[self.feature_cols].fillna(0)
        predictions = segment_model.predict(X)
        
        if hasattr(predictions, 'flatten'):
            predictions = predictions.flatten()
        
        return pd.DataFrame({
            'CUSTOMER_ID': input_df['CUSTOMER_ID'].values,
            'SEGMENT': segment,
            'PREDICTED_WEEKLY_SALES': predictions
        })

print("✅ Clase PartitionedWeeklySalesForecast definida")

# %% [markdown]
# ## 4. Crear y Registrar Modelo Particionado

# %%
if len(loaded_models) == 6:
    print("\n📝 Creando modelo particionado...\n")
    
    # Crear ModelContext con todos los modelos
    model_context = custom_model.ModelContext(
        models={
            "segment_segment_1": loaded_models['SEGMENT_1'],
            "segment_segment_2": loaded_models['SEGMENT_2'],
            "segment_segment_3": loaded_models['SEGMENT_3'],
            "segment_segment_4": loaded_models['SEGMENT_4'],
            "segment_segment_5": loaded_models['SEGMENT_5'],
            "segment_segment_6": loaded_models['SEGMENT_6'],
        }
    )
    
    partitioned_model = PartitionedWeeklySalesForecast(model_context=model_context)
    print("✅ Modelo particionado creado con 6 sub-modelos")
    
    # Sample input para schema
    training_df = session.table("ARCA_BEVERAGE_DEMO.ML_DATA.TRAINING_DATA")
    sample_input = training_df.filter(
        training_df['SEGMENT'] == 'SEGMENT_1'
    ).select([
        'CUSTOMER_ID', 'SEGMENT',
        'CUSTOMER_TOTAL_UNITS_4W', 'WEEKS_WITH_PURCHASE', 'VOLUME_QUARTILE',
        'WEEK_OF_YEAR', 'MONTH', 'QUARTER',
        'TRANSACTION_COUNT', 'UNIQUE_PRODUCTS_PURCHASED', 'AVG_UNITS_PER_TRANSACTION'
    ]).limit(5)
    
    version_date = datetime.now().strftime('%Y%m%d_%H%M')
    
    print(f"\n📝 Registrando en Model Registry...")
    print(f"   Nombre: WEEKLY_SALES_FORECAST_PARTITIONED")
    print(f"   Versión: v_{version_date}")
    
    mv = registry.log_model(
        partitioned_model,
        model_name="WEEKLY_SALES_FORECAST_PARTITIONED",
        version_name=f"v_{version_date}",
        comment="Modelo particionado creado desde 6 modelos existentes - Demo ARCA",
        metrics={
            "num_segments": 6,
            "source": "existing_models",
            "segments": ",".join(segments)
        },
        sample_input_data=sample_input,
        task=task.Task.TABULAR_REGRESSION,
        options={"function_type": "TABLE_FUNCTION"}
    )
    
    print("\n✅ Modelo registrado exitosamente!")
    
    # Set alias
    mv.set_alias("PRODUCTION")
    print(f"🏷️ Alias 'PRODUCTION' configurado")
    
else:
    print(f"\n⚠️ Solo {len(loaded_models)}/6 modelos cargados")
    print("   No se puede crear el modelo particionado")

# %% [markdown]
# ## 5. Verificar Registro

# %%
print("\n🔍 Verificando modelo particionado...\n")

result = session.sql("""
    SHOW MODELS LIKE 'WEEKLY_SALES_FORECAST_PARTITIONED' 
    IN SCHEMA ARCA_BEVERAGE_DEMO.MODEL_REGISTRY
""").collect()

if result:
    print("✅ Modelo encontrado: WEEKLY_SALES_FORECAST_PARTITIONED")
    
    versions = session.sql("""
        SHOW VERSIONS IN MODEL ARCA_BEVERAGE_DEMO.MODEL_REGISTRY.WEEKLY_SALES_FORECAST_PARTITIONED
    """).collect()
    
    print(f"\n📊 Versiones: {len(versions)}")
    for v in versions[-5:]:
        print(f"   - {v['name']}")
else:
    print("❌ Modelo no encontrado")

# %% [markdown]
# ## 6. Test de Inferencia Particionada

# %%
print("\n🧪 Inferencia particionada (VERSIÓN FINAL)...\n")

inference_sql = """
WITH test_data AS (
    SELECT 
        CUSTOMER_ID, SEGMENT,
        CUSTOMER_TOTAL_UNITS_4W, WEEKS_WITH_PURCHASE, VOLUME_QUARTILE,
        WEEK_OF_YEAR, MONTH, QUARTER,
        TRANSACTION_COUNT, UNIQUE_PRODUCTS_PURCHASED, 
        AVG_UNITS_PER_TRANSACTION::FLOAT AS AVG_UNITS_PER_TRANSACTION
    FROM ARCA_BEVERAGE_DEMO.ML_DATA.TRAINING_DATA
    QUALIFY ROW_NUMBER() OVER (PARTITION BY SEGMENT ORDER BY CUSTOMER_ID) <= 2
)
SELECT 
    p.CUSTOMER_ID,
    p.SEGMENT,
    ROUND(p.PREDICTED_WEEKLY_SALES, 2) AS PREDICTED_WEEKLY_SALES
FROM test_data t,
    TABLE(ARCA_BEVERAGE_DEMO.MODEL_REGISTRY.WEEKLY_SALES_FORECAST_PARTITIONED!PREDICT(
        t.CUSTOMER_ID, t.SEGMENT,
        t.CUSTOMER_TOTAL_UNITS_4W, t.WEEKS_WITH_PURCHASE, t.VOLUME_QUARTILE,
        t.WEEK_OF_YEAR, t.MONTH, t.QUARTER,
        t.TRANSACTION_COUNT, t.UNIQUE_PRODUCTS_PURCHASED, t.AVG_UNITS_PER_TRANSACTION
    ) OVER (PARTITION BY t.SEGMENT)) p
ORDER BY p.SEGMENT, p.CUSTOMER_ID
"""

results = session.sql(inference_sql)
results.show()

print("\n✅ ¡Modelo particionado funcionando correctamente!")
print("🎯 Una sola llamada → 6 sub-modelos → Predicciones automáticas por segmento")

# %% [markdown]
# ## Resumen
# 
# ### Lo que hicimos:
# 1. ✅ Cargamos 6 modelos existentes desde el Model Registry
# 2. ✅ Creamos UN modelo particionado que los contiene
# 3. ✅ Registramos el modelo particionado
# 4. ✅ Probamos la inferencia automática por segmento
# 
# ### Mensaje para la Demo ARCA:
# > "Los 6 modelos que entrenamos en paralelo ahora están unificados en un solo
# > modelo particionado. Una sola llamada de inferencia maneja automáticamente
# > el enrutamiento al modelo correcto según el segmento del cliente."
# 
# ### Siguiente paso:
# → Notebook `05b_partitioned_inference_true.ipynb` para demo completa de inferencia


