# Migración MLOps: Databricks → Snowflake

Este proyecto contiene la migración de un flujo MLOps completo desde Databricks hacia Snowflake.

## 📁 Estructura del Proyecto

- **`migration/`** - **Proyecto final unificado** con el flujo MLOps migrado a Snowflake
- `databricks/` - Código original de Databricks (training, inference, monitoring)
- `demo-original/` - Demos originales en formato notebook
- `demo-fine/` - Versiones refinadas de los demos

## 🎯 Proyecto Principal: `migration/`

La carpeta **`migration/`** contiene el flujo MLOps completo migrado a Snowflake, organizado en scripts secuenciales:

1. **`01_data_validation_and_cleaning.py`** - Validación y limpieza de datos
2. **`02_feature_store_setup.py`** - Configuración del Feature Store
3. **`03_hyperparameter_search.py`** - Búsqueda de hiperparámetros
4. **`04_many_model_training.py`** - Entrenamiento de modelos
5. **`05_create_partitioned_model.py`** - Creación de modelos particionados
6. **`06_partitioned_inference_batch.py`** - Inferencia por lotes particionada

## 🚀 Uso

Ejecutar los scripts en orden numérico dentro de la carpeta `migration/` para completar el flujo MLOps en Snowflake.
