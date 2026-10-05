# Migración MLOps: Databricks → Snowflake

[English](README.md) · **Español**

Migración de un flujo MLOps completo desde Databricks hacia Snowflake: validación de datos, feature store, búsqueda de hiperparámetros, entrenamiento de múltiples modelos, inferencia por lotes particionada y observabilidad del modelo (monitoreo de drift y desempeño).

## Estructura del proyecto

| Carpeta | Contenido |
|---|---|
| `migration/` | **Proyecto principal**: flujo MLOps completo migrado a Snowflake |
| `databricks/` | Código original de Databricks (training, inference, monitoring) |
| `demo-original/` | Demos originales en formato notebook |
| `demo-fine/` | Versiones refinadas de los demos |
| `arca_deployment_demo/` | Demo independiente: carga de datos, feature store, segmentación, entrenamiento de múltiples modelos y opciones de despliegue |
| `docs-js/` | Generadores de los documentos del proyecto (BBP y KT) |

## Proyecto principal: `migration/`

Scripts secuenciales, ejecutados en orden numérico. Cada `.py` tiene su notebook equivalente en `migration/notebooks/`.

### Entrenamiento

| Script | Paso |
|---|---|
| `01_data_validation_and_cleaning.py` | Validación y limpieza de los datasets de entrenamiento e inferencia |
| `02_feature_store_setup.py` | Construcción y materialización del dataset de features |
| `03_hyperparameter_search.py` | Búsqueda de hiperparámetros por grupo (LGBM / XGB) con `RandomSearch` |
| `03b_hyperparameter_search_bayesian.py` | Misma búsqueda con optimización bayesiana (`BayesOpt`) |
| `04_many_model_training.py` | Entrenamiento de un modelo por grupo (16 modelos) y registro en el Model Registry |
| `05_create_partitioned_model.py` | Integración de los 16 modelos en un único modelo particionado |

### Baselines y promoción a producción

| Script | Paso |
|---|---|
| `06a_setup_baselines.py` | Tablas auxiliares e inferencia sobre datos de entrenamiento |
| `06b_data_drift_baseline.py` | Histogramas baseline de las features de entrada |
| `06c_prediction_drift_baseline.py` | Histogramas baseline de las predicciones por segmento |
| `06d_performance_drift_baseline.py` | Métricas de desempeño baseline (WAPE, RMSE, MAE, F1) |
| `07a_copy_baselines.py` | Copia de baselines del esquema de desarrollo a producción |
| `07b_copy_models.py` | Copia de la versión `PRODUCTION` del modelo al registry de producción y etiquetado |

### Inferencia y observabilidad

| Script | Paso |
|---|---|
| `08_partitioned_inference_batch.py` | Inferencia por lotes particionada sobre datos de producción para pares (versión, semana) faltantes |
| `09a_setup_observability.py` | Tablas de destino para métricas de drift y desempeño |
| `09b_data_drift.py` | Drift de datos de entrada respecto al baseline |
| `09c_prediction_drift.py` | Drift de predicciones (Jensen-Shannon) respecto al baseline |
| `09d_performance_drift.py` | Degradación de desempeño respecto al baseline |
| `10_alertas.py` | Reporte de registros con alertas warning o critical |

`cleanup_all_resources*.sql` elimina los objetos de Snowflake creados por el flujo.

## Documentación

En `migration/docs/`: business blueprint (`documento-bbp.md`), transferencia de conocimiento (`documento-kt.md`), convención de nombres y diccionario de tablas.

## Utilidades

| Archivo | Uso |
|---|---|
| `environment.yml` | Entorno Conda |
| `convert_to_notebooks.py` | Convierte scripts `.py` (celdas `# %%`) a `.ipynb` |
| `convert_from_notebooks.py` | Convierte notebooks `.ipynb` de vuelta a `.py` |
