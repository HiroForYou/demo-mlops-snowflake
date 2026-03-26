Proyecto

Migración del Modelo de Pronóstico de Ventas de Databricks a Snowflake y Evaluación de Capacidades MLOps End-to-End - KT

KT - Versión 1.0

03 / 2025

| BUSINESS BLUEPRINT APROBACIÓN | | | |
| --- | | | | --- | --- | --- |

| Documento | Nombre | Fecha de aprobación | Firma |
| --------- | ------ | ------------------- | ----- |
|           |        |                     |       |
| ---       | ---    | ---                 | ---   |
|           |        |                     |       |
| ---       | ---    | ---                 | ---   |
|           |        |                     |       |
| ---       | ---    | ---                 | ---   |
|           |        |                     |       |
| ---       | ---    | ---                 | ---   |

**Índice**

## 1. Introducción

## 2. Arquitectura end-to-end
   - Data Preparation
   - Feature Engineering
   - Model Training
   - Baseline Generation (Training Data)
   - Environment Promotion
   - Inference (Production Data)
   - ML Observability (Production Data)
   - Alerting & Notifications

## 3. Convenciones y configuración implementada
   - Base de datos / esquemas
   - Nomenclatura de objetos
   - Gestión de versiones del modelo
   - Tablas principales
   - Parámetros Constantes

## 4. Data Preparation
### 01_data_validation_and_cleaning
   - Objetivo del Notebook
   - Código
   - Entradas
   - Salidas
   - Proceso Técnico
   - Notas Técnicas Importantes
     - Outlier Handling — P99 Threshold (Justification)
     - Audit Notes: Temporal Split + P99 Label Cleaning

## 5. Feature Store
### 02_feature_store_setup
   - Objetivo del Notebook
   - Código
   - Entradas
   - Salidas
   - Proceso Técnico

## 6. Training
### 03b_hyperparameter_search_bayesian
   - Objetivo del Notebook
   - Código
   - Entradas
   - Salidas
   - Proceso Técnico

### 04_many_model_training
   - Objetivo del Notebook
   - Código
   - Entradas
   - Salidas
   - Proceso Técnico
   - Nota Técnica: Modelo de Ejecución de MMT (Many Model Training)

### 05_create_partitioned_model
   - Objetivo del Notebook
   - Código
   - Entradas
   - Salidas
   - Proceso Técnico

## 7. Baseline Generation (Training Data)
### 06a_setup_baselines
   - Objetivo del Notebook
   - Código
   - Entradas
   - Salidas
   - Proceso Técnico

### 06b_data_drift_baseline
   - Objetivo del Notebook
   - Código
   - Entradas
   - Salidas
   - Proceso Técnico

### 06c_prediction_drift_baseline
   - Objetivo del Notebook
   - Código
   - Entradas
   - Salidas
   - Proceso Técnico

### 06d_performance_drift_baseline
   - Objetivo del Notebook
   - Código
   - Entradas
   - Salidas
   - Proceso Técnico

## 8. Environment Promotion
### 07a_copy_baselines
   - Objetivo del Notebook
   - Código
   - Entradas
   - Salidas
   - Proceso Técnico

### 07b_copy_models
   - Objetivo del Notebook
   - Código
   - Entradas
   - Salidas
   - Proceso Técnico

## 9. Inference (Production Data)
### 08_partitioned_inference_batch
   - Objetivo del Notebook
   - Código
   - Entradas
   - Salidas
   - Proceso Técnico

## 10. ML Observability (Production Data)
### 09a_setup_observability
   - Objetivo del Notebook
   - Código
   - Entradas
   - Salidas
   - Proceso Técnico

### 09b_data_drift
   - Objetivo del Notebook
   - Código
   - Entradas
   - Salidas
   - Proceso Técnico

### 09c_prediction_drift
   - Objetivo del Notebook
   - Código
   - Entradas
   - Salidas
   - Proceso Técnico

### 09d_performance_drift
   - Objetivo del Notebook
   - Código
   - Entradas
   - Salidas
   - Proceso Técnico

## 11. Alerting & Notifications
### 10_alertas
   - Objetivo del Notebook
   - Código
   - Entradas
   - Salidas
   - Proceso Técnico

# Introducción

Este documento presenta la implementación integral de un pipeline de Machine Learning desarrollado como Prueba de Concepto (POC) para la migración de Databricks a Snowflake.

El desarrollo cubre el ciclo completo de MLOps para un caso de uso de pronóstico de ventas, utilizando capacidades nativas de Snowflake y un enfoque modular basado en notebooks desacoplados.

La arquitectura fue diseñada para ser ejecutada mediante un orquestador externo, priorizando control operacional, trazabilidad y escalabilidad, sin depender de automatizaciones internas como Tasks o Dynamic Tables.

**Feature Engineering**: Se implementó un proceso estructurado de construcción de características a partir de datos históricos limpios.

El enfoque se basa en la materialización explícita de features en tablas controladas y versionadas, permitiendo trazabilidad completa sobre qué snapshot de datos fue utilizado en cada entrenamiento.

Este diseño garantiza reproducibilidad, auditoría y facilidad de integración con procesos de promoción entre ambientes.

**Training**: Se desarrolló el flujo completo de entrenamiento de modelos, incluyendo búsqueda de hiperparámetros, entrenamiento segmentado y registro formal en Snowflake Model Registry.

La arquitectura contempla un modelo por segmento de negocio y posteriormente un modelo particionado que consolida todos los submodelos en un único artefacto productivo.

El proceso permite escalabilidad, versionado controlado y capacidad de rollback por modelo o por grupo.

**Inference**: Se implementó un proceso de inferencia batch completamente ejecutado dentro de Snowflake.

En ambiente de desarrollo, el sistema consume el modelo particionado desde el Model Registry mediante alias PRODUCTION. En ambientes productivos (QA/PROD), se utilizan tags personalizados (PRODUCTION_<USE_CASE>) que permiten gestionar múltiples casos de uso de forma independiente.

El sistema ejecuta predicciones a gran escala mediante MODEL()!PREDICT con particionamiento por STATS_NTILE_GROUP y persiste los resultados en tablas estructuradas con metadatos completos (versión del modelo, entity_map, timestamps), asegurando trazabilidad operativa completa.

**Monitoring**: La arquitectura incluye un sistema completo de observabilidad MLOps que monitorea tres dimensiones críticas:

- **Data Drift**: Detecta cambios en la distribución de las features de entrada mediante histogramas y divergencia KL
- **Prediction Drift**: Monitorea cambios en la distribución de predicciones del modelo
- **Performance Drift**: Evalúa degradación del desempeño mediante métricas comparadas contra baseline

El sistema genera alertas automáticas (WARNING/CRITICAL) basadas en thresholds configurables y mantiene trazabilidad histórica completa de todas las métricas.

El desarrollo prioriza:

- Modularidad por etapas (cada notebook representa una capa del pipeline).
- Trazabilidad integral (features, hiperparámetros y modelos).
- Reproducibilidad mediante tablas materializadas y artefactos versionados.
- Escalabilidad usando capacidades nativas de Snowflake ML.
- Compatibilidad con orquestación externa empresarial.

# Arquitectura end-to-end

El pipeline MLOps implementado se estructura en siete etapas principales, cada una ejecutada mediante scripts desacoplados diseñados para ser orquestados externamente:

## Data Preparation

- **01_data_validation_and_cleaning**  
   Valida los datasets estructurados, aplica limpieza de datos (manejo de NULLs, filtrado opcional de outliers mediante umbral P99), realiza split temporal por grupo para generar conjuntos de entrenamiento y holdout, y crea tablas limpias versionadas para entrenamiento e inferencia.

## Feature Engineering

- **02_feature_store_setup**  
   Construye el dataset de features y lo materializa en una tabla controlada (sin Feature Views / sin Dynamic Tables), registrando metadatos de versión para trazabilidad. La Feature Store se organiza por entidad y frecuencia temporal (FEAT_CUSTBPR_WEEKLY), no por modelo específico.

## Model Training

- **03b_hyperparameter_search_bayesian**  
   Ejecuta búsqueda de hiperparámetros por grupo utilizando optimización bayesiana (BayesOpt) para convergencia más eficiente. Cada uno de los 16 grupos (STATS_NTILE_GROUP) se optimiza de forma independiente. Los resultados se almacenan en ML Experiments (fuente primaria) y tabla de respaldo HPO_{MODEL_NAME}. Requiere conversión de parámetros enteros a distribuciones continuas para compatibilidad con BayesOpt.

- **04_many_model_training**  
   Entrena un modelo por segmento (16 grupos esperados) mediante Many Model Training (MMT). Utiliza los mejores hiperparámetros obtenidos en la etapa anterior, registra cada modelo individual en Snowflake Model Registry con métricas completas (RMSE, MAE, WAPE, MAPE) y asigna alias PRODUCTION a las versiones entrenadas.

- **05_create_partitioned_model**  
   Construye y registra un modelo particionado (PartitionedModel) que encapsula los 16 submodelos entrenados y enruta inferencias automáticamente al modelo correcto según el valor de STATS_NTILE_GROUP. Este artefacto unificado se registra en Model Registry como UNIBOX_CUSTBPR_WEEKLY_FORECAST.

## Baseline Generation (Training Data)

- **06a_setup_baselines**  
   Inicializa la infraestructura de observabilidad, crea tablas de baseline (histogramas y métricas), ejecuta inferencia batch sobre datos de holdout usando el modelo PRODUCTION y almacena predicciones de referencia en OBS_PREDICTIONS_BL.

- **06b_data_drift_baseline**  
   Genera histogramas de referencia para todas las features del modelo utilizando el conjunto de holdout. Los histogramas se calculan por segmentos (STATS_NTILE_GROUP, CUST_CATEGORY) y se almacenan en OBS_DATA_HIST_BL.

- **06c_prediction_drift_baseline**  
   Genera histogramas de referencia para la distribución de predicciones del modelo sobre el conjunto de holdout. Se almacenan en OBS_PRED_HIST_BL para posterior comparación.

- **06d_performance_drift_baseline**  
   Calcula métricas de desempeño de referencia (WAPE, RMSE, MAE, F1_BINARY) sobre el conjunto de holdout. Las métricas base se almacenan en OBS_PERFORMANCE_BL por segmento.

## Environment Promotion

- **07a_copy_baselines**  
   Copia las tablas de baseline (histogramas de data drift, prediction drift y performance) desde el ambiente de desarrollo hacia el ambiente objetivo (QA/PROD). Sincroniza únicamente las combinaciones (MODEL_NAME, MODEL_VERSION, AGGREGATED_COL) faltantes.

- **07b_copy_models**  
   Copia los modelos registrados desde el Model Registry de desarrollo hacia el Model Registry del ambiente objetivo. Aplica tags de producción (PRODUCTION_<USE_CASE>) y rollback (ROLLBACK_VERSION_<USE_CASE>) para control de versiones operativas.

## Inference (Production Data)

- **08_partitioned_inference_batch**  
   Ejecuta inferencia batch sobre el dataset de inferencia utilizando el modelo particionado identificado mediante tags de producción (PRODUCTION_<USE_CASE>). Procesa datos por semana (WEEK), detecta combinaciones (versión, semana) faltantes en la tabla de predicciones y ejecuta MODEL()!PREDICT particionado por STATS_NTILE_GROUP. Las predicciones se almacenan en OBS_PREDICTIONS con metadatos completos (versión del modelo, timestamp, entity_map).

## ML Observability (Production Data)

- **09a_setup_observability**  
   Inicializa la infraestructura de monitoreo para producción. Crea tablas de landing para histogramas de drift (OBS_DATA_HIST, OBS_PRED_HIST), métricas de drift (OBS_DATA_DRIFT, OBS_PRED_DRIFT) y métricas de desempeño (OBS_PERFORMANCE).

- **09b_data_drift**  
   Calcula histogramas de distribución de features sobre los datos de inferencia más recientes y compara contra baseline mediante divergencia de Kullback-Leibler (KL). Detecta desviaciones significativas en las distribuciones de entrada y almacena resultados en OBS_DATA_DRIFT con niveles de alerta.

- **09c_prediction_drift**  
   Calcula histogramas de distribución de predicciones sobre inferencia reciente y compara contra baseline mediante KL divergence. Detecta cambios en el comportamiento predictivo del modelo y almacena resultados en OBS_PRED_DRIFT.

- **09d_performance_drift**  
   Calcula métricas de desempeño (WAPE, RMSE, MAE, F1_BINARY) sobre predicciones recientes mediante join con valores reales disponibles. Compara contra baseline, detecta degradación de desempeño y almacena resultados en OBS_PERFORMANCE con niveles de alerta.

## Alerting & Notifications

- **10_alertas**  
   Consolida las alertas generadas por las tablas de observabilidad (performance, data drift, prediction drift) para un modelo dado y construye un reporte unificado. El script está preparado para publicar el reporte por correo (HTML) y/o a una cola (JSON) mediante `SYSTEM$SEND_SNOWFLAKE_NOTIFICATION` (requiere Notification Integrations configuradas).

**Resultado final**:  
Un flujo completo de MLOps listo para operación batch con monitoreo continuo y gestión de alertas: datos limpios → features versionadas → HPO → entrenamiento segmentado (MMT) → modelo particionado en registry → baseline generation → promoción entre ambientes → inferencia batch productiva con trazabilidad completa → monitoreo de drift (data/prediction/performance) → consolidación y publicación de alertas.

# Convenciones y configuración implementada

### **Base de datos / esquemas**

La arquitectura utiliza una convención de esquemas por propósito funcional:

**Ambiente de Desarrollo (DEV):**
- DATABASE = "BD_AA_DEV"
- STORAGE_SCHEMA = "SC_STORAGE_BMX_PS" (datasets estructurados crudos)
- FEATURES_SCHEMA = "SC_FEATURES_BMX" (features materializadas, datasets limpios, tablas de observabilidad)
- MODELS_SCHEMA = "SC_MODELS_BMX" (Model Registry, resultados de HPO)

**Promoción a Ambientes Superiores (QA/PROD):**
- Los esquemas pueden variar según configuración del ambiente destino
- Los scripts 07a/07b manejan la sincronización de artefactos entre ambientes
- Se mantiene separación entre datos de desarrollo y producción

### **Nomenclatura de objetos**

**Feature Store:**
- Nombre base: `FEAT_CUSTBPR_WEEKLY` (organizado por entidad + frecuencia temporal)
- Sufijos por propósito:
  - `__TRAIN`: Dataset de entrenamiento limpio (≈90% temporal)
  - `__HOLDOUT`: Dataset de holdout para baseline (≈10% temporal)
  - `__INF`: Dataset de inferencia limpio
  - `__HOLDOUT_VW` / `__INF_VW`: Vistas con enriquecimiento (categorías de clientes)

**Modelo:**
- Nombre base: `UNIBOX_CUSTBPR_WEEKLY_FORECAST`
- Modelos por segmento: `{model_name}__{stats_ntile_group}` (ej: `unibox_custbpr_weekly_forecast__group_stat_0_1`)
- Modelo particionado: `UNIBOX_CUSTBPR_WEEKLY_FORECAST` (consolidado)

**Tablas de Observabilidad:**
- Baseline (DEV): `OBS_*_BL` (ej: `OBS_PREDICTIONS_BL`, `OBS_DATA_HIST_BL`, `OBS_PERFORMANCE_BL`)
- Producción (QA/PROD): `OBS_*` (ej: `OBS_PREDICTIONS`, `OBS_DATA_HIST`, `OBS_DATA_DRIFT`)

### **Gestión de versiones del modelo**

**Ambiente de Desarrollo (DEV):**
- Se utiliza el **alias PRODUCTION** para identificar la versión activa del modelo
- Sintaxis: `registry.get_model(MODEL_NAME).version("PRODUCTION")`
- El alias se asigna automáticamente en el script 04 (Many Model Training)
- Simple y directo para un único caso de uso por ambiente

**Ambientes Productivos (QA/PROD):**
- Se utilizan **tags personalizados** para gestionar múltiples casos de uso de forma independiente
- Nomenclatura: `PRODUCTION_{USE_CASE}` (ej: `PRODUCTION_CLIENTA_DEFAULT`)
- Tag de rollback: `ROLLBACK_VERSION_{USE_CASE}`
- Permite múltiples versiones productivas coexistiendo para diferentes clientes/casos de uso
- Los tags se aplican en el script 07b (Copy Models) durante la promoción

**Justificación:**
- **Alias en DEV**: Simplicidad para desarrollo iterativo de un modelo
- **Tags en PROD**: Flexibilidad para gestionar múltiples casos de uso simultáneos sin colisiones

### **Tablas principales**

**Datasets Estructurados (Entrada):**
| Tabla | Propósito | Schema |
|-------|-----------|--------|
| TRAIN_DATASET_STRUCTURED | Dataset crudo de entrenamiento | SC_STORAGE_BMX_PS |
| INFERENCE_DATASET_STRUCTURED | Dataset crudo de inferencia | SC_STORAGE_BMX_PS |
| GROUND_TRUTH_DATASET_STRUCTURED | Valores reales para evaluación | SC_STORAGE_BMX_PS |

**Feature Store (Limpieza + Transformación):**
| Tabla | Propósito | Schema |
|-------|-----------|--------|
| FEAT_CUSTBPR_WEEKLY__TRAIN | Features + labels de entrenamiento (≈90% temporal) | SC_FEATURES_BMX |
| FEAT_CUSTBPR_WEEKLY__HOLDOUT | Features + labels de holdout para baseline (≈10% temporal) | SC_FEATURES_BMX |
| FEAT_CUSTBPR_WEEKLY__INF | Features de inferencia limpias | SC_FEATURES_BMX |
| FEAT_CUSTBPR_WEEKLY | Features materializadas completas (sin target) | SC_FEATURES_BMX |

**Hiperparámetros y Entrenamiento:**
| Tabla | Propósito | Schema |
|-------|-----------|--------|
| HPO_UNIBOX_CUSTBPR_WEEKLY_FORECAST | Mejores hiperparámetros por grupo (respaldo) | SC_MODELS_BMX |
| ML Experiments | Runs de HPO con métricas (fuente primaria) | Sistema Snowflake |
| MMT_MODELS (Stage) | Artefactos de Many Model Training | SC_MODELS_BMX |

**Model Registry:**
| Modelo | Descripción | Schema |
|--------|-------------|--------|
| unibox_custbpr_weekly_forecast__group_stat_* | 16 modelos individuales por segmento | SC_MODELS_BMX |
| UNIBOX_CUSTBPR_WEEKLY_FORECAST | Modelo particionado consolidado | SC_MODELS_BMX |

**Tablas de Baseline (Generadas en DEV, scripts 06a-d):**
| Tabla | Propósito | Schema |
|-------|-----------|--------|
| OBS_PREDICTIONS_BL | Predicciones baseline sobre holdout | SC_FEATURES_BMX |
| OBS_DATA_HIST_BL | Histogramas de features baseline | SC_FEATURES_BMX |
| OBS_PRED_HIST_BL | Histogramas de predicciones baseline | SC_FEATURES_BMX |
| OBS_PERFORMANCE_BL | Métricas de desempeño baseline | SC_FEATURES_BMX |

**Tablas de Observabilidad (Producción, scripts 09a-d):**
| Tabla | Propósito | Schema |
|-------|-----------|--------|
| OBS_PREDICTIONS | Predicciones de producción con metadatos | SC_FEATURES_BMX |
| OBS_DATA_HIST | Histogramas de features en producción | SC_FEATURES_BMX |
| OBS_DATA_DRIFT | Métricas de data drift con alertas (KL divergence) | SC_FEATURES_BMX |
| OBS_PRED_HIST | Histogramas de predicciones en producción | SC_FEATURES_BMX |
| OBS_PRED_DRIFT | Métricas de prediction drift con alertas | SC_FEATURES_BMX |
| OBS_PERFORMANCE | Métricas de desempeño en producción con alertas | SC_FEATURES_BMX |

### Parámetros Constantes

| **Tipo de Objeto** | **Nombre del Objeto** | **Descripción**                            |
| ------------------ | --------------------- | ------------------------------------------ |
| Parámetro          | DATABASE              | Base de datos utilizada para el proceso (BD_AA_DEV) |
| Parámetro          | STORAGE_SCHEMA        | Schema de datasets estructurados (SC_STORAGE_BMX_PS) |
| Parámetro          | FEATURES_SCHEMA       | Schema de Feature Store y observabilidad (SC_FEATURES_BMX) |
| Parámetro          | MODELS_SCHEMA         | Schema de Model Registry y HPO (SC_MODELS_BMX) |
| Parámetro          | MODEL_NAME            | Nombre del modelo (UNIBOX_CUSTBPR_WEEKLY_FORECAST) |
| Parámetro          | FEATURE_STORE_NAME    | Nombre de Feature Store (FEAT_CUSTBPR_WEEKLY) |
| Parámetro          | TARGET_COLUMN         | Variable objetivo (UNI_BOX_WEEK) |
| Parámetro          | STATS_NTILE_GROUP_COL | Columna de segmentación (STATS_NTILE_GROUP) |
| Parámetro          | USE_CASE              | Token de caso de uso para tags en PROD (ej: CLIENTA_DEFAULT) |

# Data Preparation

### 01_data_validation_and_cleaning

Objetivo del Notebook

Este notebook tiene como objetivo validar y preparar los datasets iniciales que serán utilizados en el pipeline de entrenamiento e inferencia del modelo de Machine Learning.

El proceso realiza validaciones estructurales del dataset, limpieza de datos y generación de datasets derivados que serán utilizados en las siguientes etapas del pipeline, incluyendo el dataset limpio de entrenamiento, el dataset de inferencia limpio y un conjunto de validación holdout utilizado para monitoreo de drift.

Código

Python, Snowpark, SQL

Entradas

| **Tipo de Objeto** | **Nombre del Objeto**        | **Descripción**                                                         |
| ------------------ | ---------------------------- | ----------------------------------------------------------------------- |
| Parámetro          | Parametros Constantes        | Consultar la sección "[Parametros Constantes](#_Parametros_Constantes)" |
| Parámetro          | TARGET_COLUMN                | Variable objetivo utilizada para entrenamiento del modelo               |
| Parámetro          | STATS_NTILE_GROUP_COL        | Columna utilizada para segmentación de los datos                        |
| Parámetro          | HOLDOUT_FRACTION             | Porcentaje del dataset reservado para validación holdout                |
| Parámetro          | EXCLUDED_COLS                | Columnas que no serán consideradas como features del modelo             |
| Tabla              | TRAIN_DATASET_STRUCTURED     | Dataset estructurado inicial utilizado para entrenamiento               |
| Tabla              | INFERENCE_DATASET_STRUCTURED | Dataset estructurado utilizado para inferencia                          |

Salidas

| **Tipo de Objeto** | **Nombre del Objeto**     | **Descripción**                                                       |
| ------------------ | ------------------------- | --------------------------------------------------------------------- |
| Tabla              | FEAT_CUSTBPR_WEEKLY__TRAIN | Dataset de entrenamiento limpio (≈90% temporal) con nomenclatura basada en Feature Store |
| Tabla              | FEAT_CUSTBPR_WEEKLY__HOLDOUT | Subconjunto de validación (≈10% temporal) utilizado para monitoreo de drift |
| Tabla              | FEAT_CUSTBPR_WEEKLY__INF | Dataset de inferencia limpio utilizado para generar predicciones      |

Proceso Técnico

- Se inicializa la sesión Snowpark utilizando la función get_active_session.
- Se configuran los parámetros de entorno como base de datos, esquema y variable objetivo.
- Se consulta la tabla TRAIN_DATASET_STRUCTURED para validar la existencia del dataset y verificar su estructura.
- Se valida la presencia de la variable objetivo definida en TARGET_COLUMN.
- Se consulta el dataset de inferencia desde INFERENCE_DATASET_STRUCTURED.
- Se identifican las columnas de metadata que no deben utilizarse como features del modelo.
- Se aplica un proceso de limpieza de datos para remover registros inválidos o incompletos:
  - Filtrado de valores NULL en columnas críticas (TARGET_COLUMN, CUSTOMER_ID, WEEK, STATS_NTILE_GROUP)
  - Filtrado de valores negativos en la variable objetivo (TARGET_COLUMN >= 0)
  - Aplicación opcional de filtro de outliers mediante umbral P99 (cuando APPLY_OUTLIER_FILTER_P99=True)
- Se divide el dataset de entrenamiento en dos subconjuntos mediante split temporal por grupo:
  - **FEAT_CUSTBPR_WEEKLY__TRAIN**: contiene las semanas más antiguas (aproximadamente 90% temporal)
  - **FEAT_CUSTBPR_WEEKLY__HOLDOUT**: contiene las semanas más recientes (aproximadamente 10% temporal)
  - El split se calcula por STATS_NTILE_GROUP usando cutoff_week específico por grupo
- Los datasets generados se almacenan en:

FEAT_CUSTBPR_WEEKLY__TRAIN  
FEAT_CUSTBPR_WEEKLY__HOLDOUT  
FEAT_CUSTBPR_WEEKLY__INF

### Notas Técnicas Importantes

#### Outlier Handling — P99 Threshold (Justification)

El pipeline utiliza una estrategia robusta de limpieza de etiquetas basada en un umbral P99 del target. Esta estrategia es **opcional** y se controla mediante el parámetro `APPLY_OUTLIER_FILTER_P99` (default: False).

**Implementación sin Label Leakage:**

Cuando `APPLY_OUTLIER_FILTER_P99=True`, el proceso garantiza que no haya filtración de información (label leakage) del conjunto de holdout hacia el conjunto de entrenamiento:

1. **Primero** se calcula el `cutoff_week` temporal por grupo (basado en la distribución de registros por semana)
2. **Después** se computa el umbral P99 utilizando **únicamente los registros del conjunto TRAIN** (WEEK <= cutoff_week)
3. El umbral P99 se estima **sin utilizar información del periodo holdout** (WEEK > cutoff_week)
4. Una vez fijado el umbral, se aplica el mismo filtro consistentemente a ambos conjuntos (TRAIN y HOLDOUT) para eliminar valores extremos

Esta secuencia garantiza que la regla de limpieza no dependa de las etiquetas del conjunto holdout, preservando la integridad de la evaluación temporal.

**Justificación del umbral P99:**

- Valores por encima del percentil 99 son considerados outliers extremos que pueden distorsionar el aprendizaje del modelo
- El filtrado se aplica globalmente (no por grupo) para mantener consistencia entre segmentos
- Cuando está desactivado (APPLY_OUTLIER_FILTER_P99=False), solo se aplican filtros básicos (NULL, negativos)

#### Audit Notes: Temporal Split + P99 Label Cleaning

**¿Por qué el split no es exactamente 10%?**

El split implementado es **temporal por grupo** y se calcula utilizando periodos completos de WEEK:

- **TRAIN** utiliza `WEEK <= cutoff_week`
- **HOLDOUT** utiliza `WEEK > cutoff_week`
- El `cutoff_week` se define como la primera semana donde el share acumulativo alcanza o excede el umbral `(1 - HOLDOUT_FRACTION)`

Dado que los volúmenes se agregan por semanas completas (no por registro individual), el porcentaje resultante de holdout puede variar ligeramente del 10% configurado (por ejemplo, 9.23%). Esta variación es esperada y preserva la integridad del split temporal.

**Riesgo de Label Leakage cuando APPLY_OUTLIER_FILTER_P99=True:**

- El umbral P99 se computa **únicamente desde la ventana TRAIN** (`WEEK <= cutoff_week`)
- El periodo holdout **NO se utiliza** para estimar el umbral
- Por lo tanto, la regla de limpieza no depende de las etiquetas del holdout
- Una vez fijado el umbral, se aplica de forma consistente a ambos conjuntos (TRAIN y HOLDOUT) para eliminar valores extremos del target

Esta implementación garantiza que no exista filtración de información entre los conjuntos de datos.

# Feature Store

### 02_feature_store_setup

Objetivo del Notebook

Este notebook construye la tabla de **Feature Store** que será utilizada para entrenar los modelos de Machine Learning.

El proceso identifica las columnas que representan features del modelo, excluyendo columnas de metadata y la variable objetivo, y crea una tabla consolidada que será utilizada en las etapas de entrenamiento.

Código

Python, Snowpark, SQL

Entradas

| **Tipo de Objeto** | **Nombre del Objeto** | **Descripción**                                                         |
| ------------------ | --------------------- | ----------------------------------------------------------------------- |
| Parámetro          | Parámetros Constantes | Consultar la sección "[Parametros Constantes](#_Parametros_Constantes)" |
| Parámetro          | TARGET_COLUMN         | Variable objetivo utilizada en el modelo (UNI_BOX_WEEK)                 |
| Parámetro          | STATS_NTILE_GROUP_COL | Columna de segmentación utilizada para entrenamiento                    |
| Parámetro          | EXCLUDED_COLS         | Columnas excluidas del conjunto de features                             |
| Tabla              | FEAT_CUSTBPR_WEEKLY__TRAIN | Dataset limpio utilizado para generar las features                 |

Salidas

| **Tipo de Objeto** | **Nombre del Objeto** | **Descripción**                                                             |
| ------------------ | --------------------- | --------------------------------------------------------------------------- |
| Tabla              | FEAT_CUSTBPR_WEEKLY   | Tabla de Feature Store que contiene las features utilizadas por los modelos |

Proceso Técnico

- Se inicializa la sesión Snowpark.
- Se consulta la estructura de la tabla FEAT_CUSTBPR_WEEKLY__TRAIN utilizando DESCRIBE TABLE.
- Se identifican todas las columnas disponibles en el dataset.
- Se eliminan las columnas definidas en EXCLUDED_COLS (CUSTOMER_ID, BRAND_PRES_RET, PROD_KEY, WEEK, STATS_NTILE_GROUP).
- Las columnas restantes se consideran features del modelo.
- Se construye un dataset que incluye:

Columnas de identificación: CUSTOMER_ID, BRAND_PRES_RET, PROD_KEY  
Columna temporal: WEEK  
Columna de segmentación: STATS_NTILE_GROUP  
Features del modelo (todas las columnas numéricas restantes)

- El dataset resultante se almacena en la tabla FEAT_CUSTBPR_WEEKLY.
- Esta tabla se utiliza posteriormente en los scripts de entrenamiento (03b, 04) para construir el dataset completo mediante join con las labels.

# Training

### 03b_hyperparameter_search_bayesian

Objetivo del Notebook

Este notebook ejecuta el proceso de **búsqueda de hiperparámetros utilizando optimización bayesiana** para identificar las configuraciones óptimas de los modelos de Machine Learning.

El proceso evalúa múltiples combinaciones de hiperparámetros para cada segmento de datos y registra los resultados obtenidos.

Código

Python, Snowpark, Snowflake ML Tuner, Bayesian Optimization

Entradas

| **Tipo de Objeto** | **Nombre del Objeto** | **Descripción**                                                         |
| ------------------ | --------------------- | ----------------------------------------------------------------------- |
| Parámetro          | Parametros Constantes | Consultar la sección "[Parametros Constantes](#_Parametros_Constantes)" |
| Parámetro          | TARGET_COLUMN         | Variable objetivo del modelo (UNI_BOX_WEEK)                             |
| Parámetro          | STATS_NTILE_GROUP_COL | Columna utilizada para segmentar los modelos                            |
| Parámetro          | NUM_TRIALS            | Número de pruebas de hiperparámetros (default: 15)                      |
| Parámetro          | MAX_CONCURRENT_TRIALS | Número máximo de ejecuciones paralelas (default: 4)                     |
| Parámetro          | SAMPLE_RATE_PER_GROUP | Fracción de datos utilizada durante el tuning (default: 0.2)            |
| Tabla              | FEAT_CUSTBPR_WEEKLY__TRAIN | Dataset de entrenamiento limpio                                    |
| Tabla              | FEAT_CUSTBPR_WEEKLY      | Feature Store utilizada para entrenamiento                           |

Salidas

| **Tipo de Objeto** | **Nombre del Objeto**  | **Descripción**                                                                  |
| ------------------ | ---------------------- | -------------------------------------------------------------------------------- |
| Tabla              | HPO_UNIBOX_CUSTBPR_WEEKLY_FORECAST | Tabla que almacena los resultados del proceso de optimización de hiperparámetros (respaldo) |
| Experimento ML     | EXP_UNIBOX_CUSTBPR_WEEKLY_FORECAST_BAYESIAN_{DATE} | Experimento de Snowflake ML que contiene runs de tuning por grupo (fuente primaria) |

Proceso Técnico

- Se inicializa la sesión Snowpark.
- Se configuran los parámetros del experimento de optimización.
- Se define el espacio de búsqueda de hiperparámetros:
  - Todos los parámetros usan distribuciones continuas `uniform()` (requerido por BayesOpt)
  - Parámetros enteros se convierten dentro de la función de entrenamiento
- Se escala el cluster de ejecución (CLUSTER_SIZE_HPO nodos).
- Para cada segmento de datos (STATS_NTILE_GROUP):
  - Se carga un subconjunto de datos (SAMPLE_RATE_PER_GROUP, default 20%)
  - Se utiliza el algoritmo de **Bayesian Optimization** para explorar el espacio de hiperparámetros de forma eficiente
  - Se ejecutan NUM_TRIALS pruebas de entrenamiento con MAX_CONCURRENT_TRIALS ejecuciones en paralelo
  - Se evalúa cada configuración mediante RMSE en un split temporal (80% train, 20% validation)
  - Se registran las métricas obtenidas durante cada prueba en ML Experiments (fuente primaria)
- Los mejores hiperparámetros por grupo se persisten en la tabla HPO_UNIBOX_CUSTBPR_WEEKLY_FORECAST como respaldo.
- Se escala el cluster de vuelta a tamaño mínimo.

### Notas técnicas: Modelos, BayesOpt y Search Space (03b)

#### ¿Por qué se usan XGBRegressor y LGBMRegressor?

Se seleccionan dos familias de **Gradient Boosted Trees** por su desempeño típico en regresión tabular con no-linealidades e interacciones complejas, su robustez ante escalas heterogéneas y su buen trade-off entre performance y costo computacional:

- **XGBRegressor**: suele ser estable y fuerte con regularización explícita (L1/L2) y control fino de complejidad por árbol.
- **LGBMRegressor**: suele ser eficiente en entrenamiento y puede capturar relaciones complejas con `num_leaves` (estructura de hojas) y controles de regularización.

Adicionalmente, en Snowflake se usan wrappers `snowflake.ml.modeling.*` para compatibilidad con el runtime (serialización/ejecución remota) y el ecosistema de Tuner/MMT.

#### Resumen breve: ¿cómo funciona la búsqueda bayesiana (BayesOpt)?

BayesOpt es un método **secuencial** que, en cada iteración, propone el siguiente conjunto de hiperparámetros a evaluar usando un modelo sustituto (“surrogate”) del rendimiento en función de los hiperparámetros y una función de adquisición. En este script se usa:

- `BayesOpt(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})`

La variante **UCB (Upper Confidence Bound)** balancea:

- **Explotación**: probar zonas con buen RMSE esperado.
- **Exploración**: probar zonas con incertidumbre alta (controlado por `kappa`).

#### Search Space (rangos) por modelo

Todos los parámetros se definen como `uniform(min, max)` por requerimiento de BayesOpt (continuo). Los parámetros enteros se muestrean como float y se castean a int en la función de entrenamiento (INT_PARAMS).

**XGBRegressor** (uniform):  
| Hiperparámetro | Rango | Rol técnico |
| --- | --- | --- |
| `n_estimators` | 50–300 | Nº de árboles; controla capacidad y costo |
| `max_depth` | 3–10 | Profundidad por árbol; controla complejidad |
| `learning_rate` | 0.01–0.3 | Tasa de aprendizaje; trade-off con nº de árboles |
| `subsample` | 0.6–1.0 | Submuestreo de filas; regularización/robustez |
| `colsample_bytree` | 0.6–1.0 | Submuestreo de columnas; reduce overfit |
| `min_child_weight` | 1–7 | Controla splits por soporte de datos (regularización) |
| `gamma` | 0–0.5 | Penaliza splits; reduce complejidad |
| `reg_alpha` | 0–1 | Regularización L1; sparsity/robustez |
| `reg_lambda` | 0–1 | Regularización L2; estabilidad |

**LGBMRegressor** (uniform):  
| Hiperparámetro | Rango | Rol técnico |
| --- | --- | --- |
| `n_estimators` | 50–300 | Nº de boosting iterations |
| `max_depth` | 3–10 | Límite de profundidad; controla complejidad |
| `learning_rate` | 0.01–0.3 | Tasa de aprendizaje |
| `num_leaves` | 20–150 | Complejidad del árbol (hojas); controla capacidad |
| `subsample` | 0.6–1.0 | Bagging de filas; regularización |
| `colsample_bytree` | 0.6–1.0 | Bagging de columnas; regularización |
| `reg_alpha` | 0–1 | Regularización L1 |
| `reg_lambda` | 0–1 | Regularización L2 |
| `min_child_samples` | 5–50 | Mín. muestras por hoja; evita sobreajuste en hojas pequeñas |

### 04_many_model_training

Objetivo del Notebook

Este notebook ejecuta el proceso de entrenamiento de múltiples modelos utilizando la funcionalidad **Many Model Training (MMT)** de Snowflake.

Se entrena un modelo independiente para cada segmento definido por la columna STATS_NTILE_GROUP.

Código

Python, Snowpark, Snowflake ML ManyModelTraining

Entradas

| **Tipo de Objeto** | **Nombre del Objeto**  | **Descripción**                                                         |
| ------------------ | ---------------------- | ----------------------------------------------------------------------- |
| Parámetro          | Parametros Constantes  | Consultar la sección "[Parametros Constantes](#_Parametros_Constantes)" |
| Parámetro          | TARGET_COLUMN          | Variable objetivo del modelo (UNI_BOX_WEEK)                             |
| Parámetro          | STATS_NTILE_GROUP_COL  | Columna utilizada para segmentar los modelos                            |
| Parámetro          | GROUP_MODEL            | Mapeo entre segmento y algoritmo de ML (LGBM/XGB por grupo)             |
| Tabla              | FEAT_CUSTBPR_WEEKLY__TRAIN | Dataset de entrenamiento limpio                                     |
| Tabla              | FEAT_CUSTBPR_WEEKLY       | Feature Store utilizada para entrenamiento                           |
| Tabla              | HPO_UNIBOX_CUSTBPR_WEEKLY_FORECAST | Resultados de optimización de hiperparámetros (respaldo)      |
| Experimento ML     | EXP_UNIBOX_CUSTBPR_WEEKLY_FORECAST_BAYESIAN_{DATE} | ML Experiments con hiperparámetros (fuente primaria) |

Salidas

| **Tipo de Objeto** | **Nombre del Objeto** | **Descripción**                                              |
| ------------------ | --------------------- | ------------------------------------------------------------ |
| Stage Snowflake    | MMT_MODELS            | Stage donde se almacenan los modelos entrenados por segmento |
| Model Registry     | unibox_custbpr_weekly_forecast__group_stat_* | 16 modelos registrados individualmente por grupo (con alias PRODUCTION) |

Proceso Técnico

- Se inicializa la sesión Snowpark.
- Se consulta la Feature Store FEAT_CUSTBPR_WEEKLY.
- Se recuperan los mejores hiperparámetros desde HPO_UNIBOX_CUSTBPR_WEEKLY_FORECAST (respaldo) o desde ML Experiments EXP_UNIBOX_CUSTBPR_WEEKLY_FORECAST_BAYESIAN_{DATE} (fuente primaria).
- Se cargan los hiperparámetros por grupo, utilizando valores por defecto cuando no están disponibles.
- Se ejecuta el proceso Many Model Training (MMT):
  - Configuración del cluster de ejecución con escalado dinámico (CLUSTER_SIZE_MMT nodos)
  - Particionamiento de datos por STATS_NTILE_GROUP
  - Ejecución distribuida mediante Ray/Snowpark-managed runtime
  - Para cada grupo se entrena un modelo independiente utilizando el algoritmo asignado (LGBMRegressor o XGBRegressor según GROUP_MODEL)
  - Cada función de entrenamiento por partición ejecuta:
    - Join entre FEAT_CUSTBPR_WEEKLY (features) y FEAT_CUSTBPR_WEEKLY__TRAIN (labels)
    - Split temporal interno (80% train, 20% test por grupo)
    - Conversión a matrices numéricas
    - Entrenamiento del regresor con hiperparámetros optimizados
    - Evaluación de métricas (RMSE, MAE, WAPE, MAPE)
    - Adjunción de metadatos al modelo (métricas, hiperparámetros, columnas de features)
- Los modelos entrenados se almacenan en el stage MMT_MODELS.
- Cada modelo se registra individualmente en Snowflake Model Registry:
  - Nombre del modelo: `unibox_custbpr_weekly_forecast__{group_name}` (ejemplo: `unibox_custbpr_weekly_forecast__group_stat_0_1`)
  - Versión: `v_{VERSION_DATE}`
  - Métricas registradas: RMSE, MAE, WAPE, MAPE, training_samples, test_samples, algorithm, hyperparameter_search_id, feature_table_name
  - Task type: TABULAR_REGRESSION
- Se asigna el alias PRODUCTION a cada versión recién entrenada (usado en DEV).
- Al finalizar, se escala el cluster de vuelta a tamaño mínimo.

### Nota técnica: Normalización y casteo de hiperparámetros (04)

Los hiperparámetros recuperados desde Experiments/tabla pueden llegar con tipos no nativos (por ejemplo, `numpy.float64`). Antes de instanciar el estimador, el script:

- Define un set de parámetros enteros esperados (ej: `n_estimators`, `max_depth`, `num_leaves`, `min_child_weight`, `min_child_samples`).
- Castea a `int` los parámetros del set entero y a `float` los continuos cuando aplica.
- En caso de error de casteo, cae a defaults por modelo (DEFAULT_PARAMS_BY_MODEL) para mantener robustez del pipeline.

Esto evita fallas por tipos y asegura que los constructores de `XGBRegressor`/`LGBMRegressor` reciban valores válidos.

### Nota Técnica: Modelo de Ejecución de MMT (Many Model Training)

Una preocupación común al ver imports de `snowflake.ml.modeling.*` dentro de la función de entrenamiento es si estamos instanciando "un modelo distribuido dentro de un nodo", creando una distribución anidada problemática.

**Clarificación del modelo de ejecución:**

- **`ManyModelTraining` es el sistema distribuido**: particiona el dataset de entrada (por ejemplo, por `STATS_NTILE_GROUP`) y programa **una partición por worker** en el runtime administrado por Snowflake.

- **Dentro de cada worker**, la función de entrenamiento opera sobre **un pandas DataFrame** que contiene únicamente los datos de esa partición. El entrenamiento ejecutado es **completamente local a esa partición** — no hay entrenamiento distribuido anidado dentro del estimador.

- **Los estimadores en `snowflake.ml.modeling.*` son wrappers compatibles con el runtime**: están diseñados para funcionar de forma confiable dentro del entorno de ejecución de Snowflake (gestión de dependencias, serialización, ejecución remota). Utilizar librerías nativas directamente (por ejemplo, `xgboost.XGBRegressor`) puede funcionar en algunos setups, pero puede romper la serialización o el empaquetado en el runtime de Snowflake; por eso se recomiendan los wrappers oficiales para MMT.

- **Cada modelo entrenado es independiente**: cada `STATS_NTILE_GROUP` produce un artefacto de modelo separado que se registra individualmente en Model Registry. No existe coordinación entre los entrenamientos de diferentes grupos durante la fase de MMT.

**En resumen:**  
`ManyModelTraining` distribuye el trabajo de entrenamiento **entre particiones**, y cada partición entrena **un modelo local estándar** sobre su subconjunto de datos. No hay distribución anidada ni entrenamiento distribuido dentro de los estimadores individuales.

### 05_create_partitioned_model

Objetivo del Notebook

Este notebook crea un **modelo particionado** que agrupa los modelos entrenados por segmento en un único modelo lógico.

Este modelo actúa como un wrapper que selecciona automáticamente el modelo correspondiente dependiendo del valor de la columna de segmentación.

Código

Python, Snowpark, Snowflake ML Registry

Entradas

| **Tipo de Objeto** | **Nombre del Objeto**  | **Descripción**                                                         |
| ------------------ | ---------------------- | ----------------------------------------------------------------------- |
| Parámetro          | Parametros Constantes  | Consultar la sección "[Parametros Constantes](#_Parametros_Constantes)" |
| Parámetro          | MODEL_NAME             | Nombre del modelo particionado (UNIBOX_CUSTBPR_WEEKLY_FORECAST)         |
| Parámetro          | STATS_NTILE_GROUP_COL  | Columna utilizada para seleccionar el modelo (STATS_NTILE_GROUP)        |
| Tabla              | FEAT_CUSTBPR_WEEKLY__TRAIN | Dataset utilizado para identificar los grupos de entrenamiento      |
| Model Registry     | unibox_custbpr_weekly_forecast__group_stat_* | 16 modelos previamente entrenados (con alias PRODUCTION) |

Salidas

| **Tipo de Objeto** | **Nombre del Objeto**          | **Descripción**                                               |
| ------------------ | ------------------------------ | ------------------------------------------------------------- |
| Model Registry     | UNIBOX_CUSTBPR_WEEKLY_FORECAST | Modelo particionado que agrupa todos los modelos por segmento (con alias PRODUCTION en DEV) |

Proceso Técnico

- Se inicializa la sesión Snowpark.
- Se consulta el dataset FEAT_CUSTBPR_WEEKLY__TRAIN para identificar los segmentos existentes (16 grupos STATS_NTILE_GROUP).
- Para cada grupo definido en STATS_NTILE_GROUP se carga el modelo correspondiente desde el Model Registry utilizando el alias PRODUCTION.
- Se construye una estructura que mapea cada grupo con su modelo correspondiente.
- Se define una clase CustomModel (PartitionedModel) que:
  - Encapsula los 16 submodelos en un diccionario internal
  - Implementa el método `predict()` que enruta automáticamente cada fila al submodelo correcto según STATS_NTILE_GROUP
  - Incluye columnas contextuales (CUSTOMER_ID, WEEK, BRAND_PRES_RET, PROD_KEY) en el output
- El modelo particionado resultante se registra en Snowflake Model Registry con:
  - Nombre: UNIBOX_CUSTBPR_WEEKLY_FORECAST
  - Version: v_{VERSION_DATE}
  - Alias: PRODUCTION (en DEV)
  - Task: TABULAR_REGRESSION
- Este modelo unificado permite inferencia simplificada: un único llamado a MODEL()!PREDICT maneja automáticamente el enrutamiento interno.

# Baseline Generation (Training Data)

Esta etapa genera los baselines de referencia utilizando el conjunto de holdout del entrenamiento. Los baselines se utilizarán posteriormente para detectar drift en producción.

### 06a_setup_baselines

Objetivo del Notebook

Este notebook inicializa la infraestructura de observabilidad y ejecuta inferencia batch sobre el conjunto de holdout para generar predicciones de referencia (baseline).

El notebook crea las tablas necesarias para almacenar histogramas y métricas de baseline, carga el modelo PRODUCTION desde Model Registry, ejecuta inferencia particionada sobre los datos de holdout y almacena las predicciones que servirán como referencia para los procesos de monitoreo de drift.

Código

Python, Snowpark, SQL, Snowflake ML Model Registry

Entradas

| **Tipo de Objeto** | **Nombre del Objeto**      | **Descripción**                                                                          |
| ------------------ | -------------------------- | ---------------------------------------------------------------------------------------- |
| Parámetro          | Parametros Constantes      | Consultar la sección "[Parametros Constantes](#_Parametros_Constantes)"                  |
| Parámetro          | MODEL_NAME                 | Nombre del modelo registrado en Snowflake Model Registry                                 |
| Parámetro          | PARTITION_COL              | Columna utilizada para particionar el modelo (STATS_NTILE_GROUP)                         |
| Parámetro          | TARGET_COL                 | Variable objetivo utilizada durante entrenamiento (UNI_BOX_WEEK)                         |
| Parámetro          | PREDICTION_COL             | Columna donde se almacenará la predicción (PREDICTED_UNI_BOX_WEEK)                       |
| Parámetro          | TIME_COL                   | Columna temporal utilizada para batching (week)                                          |
| Parámetro          | BASELINE_ALIAS             | Alias del modelo en registry (PRODUCTION)                                                |
| Parámetro          | N_BINS                     | Número de bins para histogramas de drift (default: 20)                                   |
| Tabla              | FEAT_CUSTBPR_WEEKLY__HOLDOUT | Dataset de holdout para generar baseline                                               |
| Tabla              | TRAIN_CUST_CATEGORY_LOOKUP | Tabla de referencia de categorías de clientes                                            |
| Servicio Snowflake | Snowflake Model Registry   | Repositorio donde se encuentra el modelo PRODUCTION                                      |

Salidas

| **Tipo de Objeto** | **Nombre del Objeto**         | **Descripción**                                                                      |
| ------------------ | ----------------------------- | ------------------------------------------------------------------------------------ |
| Tabla              | OBS_PREDICTIONS_BL            | Tabla que almacena las predicciones baseline con metadatos completos                 |
| Tabla              | OBS_PREDICTIONS_BL_VW         | Vista transient que join predictions con categorías de clientes                      |
| Tabla              | OBS_DATA_HIST_BL              | Tabla para almacenar histogramas de features baseline (creada, poblada en 06b)       |
| Tabla              | OBS_PRED_HIST_BL              | Tabla para almacenar histogramas de predicciones baseline (creada, poblada en 06c)   |
| Tabla              | OBS_PERFORMANCE_BL            | Tabla para almacenar métricas de desempeño baseline (creada, poblada en 06d)         |

Proceso Técnico

- Se inicializa la sesión Snowpark mediante get_active_session.
- Se crean las tablas de baseline si no existen:
  - OBS_DATA_HIST_BL: para histogramas de data drift
  - OBS_PRED_HIST_BL: para histogramas de prediction drift
  - OBS_PERFORMANCE_BL: para métricas de performance
  - OBS_PREDICTIONS_BL: para predicciones de referencia
- Se crea la lookup table TRAIN_CUST_CATEGORY_LOOKUP para mapear categorías de clientes.
- Se crea la vista FEAT_CUSTBPR_WEEKLY__HOLDOUT_VW combinando holdout data con categorías.
- Se conecta al Model Registry y se obtiene la versión asociada al alias PRODUCTION.
- Se verifica si ya existen predicciones baseline para esta versión del modelo.
- Si no existen predicciones previas:
  - Se itera sobre cada valor único de TIME_COL (week)
  - Para cada batch temporal se ejecuta MODEL()!PREDICT con particionamiento por STATS_NTILE_GROUP
  - Las predicciones se insertan en OBS_PREDICTIONS_BL con:
    - RECORD_ID (hash único por registro)
    - MODEL_NAME y MODEL_VERSION
    - ENTITY_MAP (objeto JSON con metadatos del registro)
    - PREDICTION (valor predicho)
    - BKCC, CALMONTH, LDTS (metadatos operativos)
- Se crea la tabla transient OBS_PREDICTIONS_BL_VW mediante join con categorías de clientes.
- Las predicciones almacenadas servirán como referencia para los scripts 06b, 06c y 06d.

### 06b_data_drift_baseline

Objetivo del Notebook

Este notebook genera el **baseline estadístico de las variables de entrada utilizadas por el modelo**, el cual será utilizado posteriormente para detectar **data drift** durante la operación del modelo en producción.

Código que usa el notebook

Lenguajes utilizados:

Python  
Snowpark  
SQL

Entradas

| **Tipo de Objeto** | **Nombre del Objeto**      | **Descripción**                                                                            |
| ------------------ | -------------------------- | ------------------------------------------------------------------------------------------ |
| Parámetro          | Parametros Constantes      | Consultar la sección "[Parametros Constantes](#_Parametros_Constantes)"                    |
| Parámetro          | PARTITION_COL              | Columna utilizada para segmentar los datos durante el análisis                             |
| Parámetro          | TARGET_COL                 | Variable objetivo utilizada en el modelo                                                   |
| Parámetro          | TIME_COL                   | Columna temporal utilizada para identificar periodos                                       |
| Parámetro          | N_BINS                     | Número de bins utilizados para generar histogramas de distribución                         |
| View               | TRAIN_DATASET_HOLDOUT_VW   | Vista que contiene el dataset de referencia utilizado para generar el baseline de features |
| View               | DA_PREDICTIONS_BASELINE_VW | Vista que contiene las predicciones generadas por el modelo                                |

Salidas

| **Tipo de Objeto** | **Nombre del Objeto**             | **Descripción**                                                                                                  |
| ------------------ | --------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Tabla              | DA_DATA_DRIFT_HISTOGRAMS_BASELINE | Tabla que contiene los histogramas base utilizados para detectar desviaciones en la distribución de las features |

Proceso Técnico

- Se inicializa la sesión Snowpark.
- Se consulta el dataset de referencia desde TRAIN_DATASET_HOLDOUT_VW.
- Se identifican las variables utilizadas por el modelo que deben ser monitoreadas para detectar data drift.
- Se excluyen columnas que no deben participar en el análisis de drift.
- Para cada feature se calculan histogramas de distribución utilizando un número definido de bins.
- Los histogramas se calculan por segmentos definidos por STATS_NTILE_GROUP y CUST_CATEGORY.
- Los resultados se almacenan en la tabla DA_DATA_DRIFT_HISTOGRAMS_BASELINE.

### 06c_prediction_drift_baseline

Objetivo del Notebook

Este notebook genera el **baseline de la distribución de las predicciones del modelo** utilizando las predicciones almacenadas por el script 06a, el cual será utilizado para detectar **prediction drift** durante la operación del modelo en producción.

Código que usa el notebook

Lenguajes utilizados:

Python  
Snowpark  
SQL

Entradas

| **Tipo de Objeto** | **Nombre del Objeto**      | **Descripción**                                                         |
| ------------------ | -------------------------- | ----------------------------------------------------------------------- |
| Parámetro          | Parametros Constantes      | Consultar la sección "[Parametros Constantes](#_Parametros_Constantes)" |
| Parámetro          | PREDICTION_COL             | Columna que contiene las predicciones (PREDICTED_UNI_BOX_WEEK)          |
| Parámetro          | PARTITION_COL              | Columna utilizada para segmentar los datos (STATS_NTILE_GROUP)          |
| Parámetro          | TIME_COL                   | Columna temporal utilizada en el análisis (week)                        |
| Parámetro          | N_BINS                     | Número de bins utilizados para construir histogramas (default: 20)      |
| Tabla              | OBS_PREDICTIONS_BL_VW      | Vista que contiene las predicciones baseline generadas por 06a          |

Salidas

| **Tipo de Objeto** | **Nombre del Objeto**                   | **Descripción**                                                                |
| ------------------ | --------------------------------------- | ------------------------------------------------------------------------------ |
| Tabla              | OBS_PRED_HIST_BL                        | Tabla que contiene los histogramas base de la distribución de las predicciones |

Proceso Técnico

- Se inicializa la sesión Snowpark.
- Se consulta la vista OBS_PREDICTIONS_BL_VW que contiene las predicciones de referencia.
- Se identifican todas las combinaciones únicas de (MODEL_NAME, MODEL_VERSION) presentes en las predicciones baseline.
- Para cada combinación:
  - Se agrupan las predicciones por segmentos (STATS_NTILE_GROUP, CUST_CATEGORY)
  - Se generan histogramas de distribución de predicciones utilizando N_BINS bins
  - Los histogramas se construyen mediante WIDTH_BUCKET para crear bins equiespaciados
  - Se calcula el conteo de predicciones por bin y se normaliza para obtener frecuencias relativas
- Cada histograma se almacena en OBS_PRED_HIST_BL con:
  - RECORD_ID: hash único por combinación
  - MODEL_NAME y MODEL_VERSION
  - ENTITY_MAP: metadatos de la agregación
  - AGGREGATED_COL y AGGREGATED_VALUE: dimensiones de segmentación
  - METRIC_COL: siempre "PREDICTED_UNI_BOX_WEEK"
  - METRIC_MAP: objeto JSON con el histograma (bin_edges y frequencies)
  - CALMONTH y LDTS: metadatos temporales
- Los histogramas almacenados servirán como referencia para detectar desviaciones en la distribución de predicciones en producción.

### 06d_performance_drift_baseline

Objetivo del Notebook

Este notebook genera el **baseline de métricas de desempeño del modelo** utilizando las predicciones baseline y los valores reales del conjunto de holdout, el cual será utilizado para detectar **performance drift** durante la operación del modelo en producción.

Código que usa el notebook

Lenguajes utilizados:

Python  
Snowpark  
SQL

Entradas

| **Tipo de Objeto** | **Nombre del Objeto**         | **Descripción**                                                         |
| ------------------ | ----------------------------- | ----------------------------------------------------------------------- |
| Parámetro          | Parametros Constantes         | Consultar la sección "Parametros Constantes"                            |
| Parámetro          | TARGET_COL                    | Variable objetivo utilizada para evaluar el modelo (UNI_BOX_WEEK)       |
| Parámetro          | PREDICTION_COL                | Columna que contiene las predicciones (PREDICTED_UNI_BOX_WEEK)          |
| Parámetro          | PARTITION_COL                 | Columna utilizada para segmentar el análisis (STATS_NTILE_GROUP)        |
| Parámetro          | TIME_COL                      | Columna temporal utilizada para evaluación (week)                       |
| Tabla              | FEAT_CUSTBPR_WEEKLY__HOLDOUT_VW | Vista que contiene los valores reales del conjunto de holdout         |
| Tabla              | OBS_PREDICTIONS_BL_VW         | Vista que contiene las predicciones baseline generadas por 06a          |

Salidas

| **Tipo de Objeto** | **Nombre del Objeto**   | **Descripción**                                              |
| ------------------ | ----------------------- | ------------------------------------------------------------ |
| Tabla              | OBS_PERFORMANCE_BL      | Tabla que almacena las métricas base de desempeño del modelo |

Proceso Técnico

- Se inicializa la sesión Snowpark.
- Se consultan los valores reales (actuals) desde FEAT_CUSTBPR_WEEKLY__HOLDOUT_VW.
- Se consultan las predicciones baseline desde OBS_PREDICTIONS_BL_VW.
- Se realiza un join entre actuals y predictions utilizando las claves:
  - customer_id
  - brand_pres_ret
  - prod_key
  - week
- Para cada combinación de (MODEL_NAME, MODEL_VERSION, STATS_NTILE_GROUP, CUST_CATEGORY) se calculan métricas de desempeño:
  - **WAPE** (Weighted Absolute Percentage Error): suma de errores absolutos dividida por suma de valores reales
  - **RMSE** (Root Mean Squared Error): raíz cuadrada del error cuadrático medio
  - **MAE** (Mean Absolute Error): promedio de errores absolutos
  - **F1_BINARY**: métrica binaria basada en threshold (target > 0)
- Las métricas se almacenan en OBS_PERFORMANCE_BL con:
  - RECORD_ID: hash único por combinación
  - MODEL_NAME y MODEL_VERSION
  - ENTITY_MAP: metadatos de la agregación
  - AGGREGATED_COL y AGGREGATED_VALUE: dimensiones de segmentación
  - METRIC_COL: nombre de la métrica
  - METRIC_VALUE: valor calculado de la métrica
  - BKCC, CALMONTH, LDTS: metadatos operativos
- Las métricas baseline servirán como referencia para detectar degradación del modelo en producción.

# Environment Promotion

Esta etapa sincroniza artefactos entre ambientes (DEV → QA/PROD), copiando baselines y modelos registrados para preparar el ambiente objetivo.

### 07a_copy_baselines

Objetivo del Notebook

Copiar las tablas de baseline (histogramas y métricas de referencia) desde el ambiente de desarrollo hacia el ambiente objetivo (QA o PROD). Este script sincroniza únicamente los registros faltantes, evitando duplicados.

Código

Python, Snowpark, SQL

Entradas

| **Tipo de Objeto** | **Nombre del Objeto** | **Descripción**                                           |
| ------------------ | --------------------- | --------------------------------------------------------- |
| Parámetro          | SRC_DATABASE          | Base de datos fuente (desarrollo)                         |
| Parámetro          | TGT_DATABASE          | Base de datos destino (QA/PROD)                           |
| Parámetro          | SRC_STORAGE_SCHEMA    | Schema fuente donde residen los baselines                 |
| Parámetro          | TGT_STORAGE_SCHEMA    | Schema destino donde se copiarán los baselines            |
| Parámetro          | MODEL_NAME            | Nombre del modelo para filtrar registros a copiar         |
| Tabla              | OBS_DATA_HIST_BL (SRC) | Tabla fuente de histogramas de data drift baseline       |
| Tabla              | OBS_PRED_HIST_BL (SRC) | Tabla fuente de histogramas de prediction drift baseline |
| Tabla              | OBS_PERFORMANCE_BL (SRC) | Tabla fuente de métricas de performance baseline       |

Salidas

| **Tipo de Objeto** | **Nombre del Objeto**        | **Descripción**                                          |
| ------------------ | ---------------------------- | -------------------------------------------------------- |
| Tabla              | OBS_DATA_HIST_BL (TGT)       | Tabla destino de histogramas de data drift baseline      |
| Tabla              | OBS_PRED_HIST_BL (TGT)       | Tabla destino de histogramas de prediction drift baseline|
| Tabla              | OBS_PERFORMANCE_BL (TGT)     | Tabla destino de métricas de performance baseline        |

Proceso Técnico

- Se inicializa la sesión Snowpark.
- Para cada par de tablas (fuente, destino):
  - Se crea la tabla destino si no existe (usando CREATE TABLE ... LIKE)
  - Se identifican combinaciones (MODEL_NAME, MODEL_VERSION, AGGREGATED_COL) que existen en fuente pero no en destino
  - Se copian únicamente los registros faltantes mediante INSERT INTO ... SELECT con filtro de exclusión
  - Se reporta el número de registros copiados
- Las tablas sincronizadas son:
  - Histogramas de data drift baseline (OBS_DATA_HIST_BL)
  - Histogramas de prediction drift baseline (OBS_PRED_HIST_BL)
  - Métricas de performance baseline (OBS_PERFORMANCE_BL)
- Este script NO copia los modelos del registry (eso lo hace 07b).

### 07b_copy_models

Objetivo del Notebook

Copiar los modelos registrados desde el Model Registry de desarrollo hacia el Model Registry del ambiente objetivo (QA o PROD), aplicando tags de producción para control de versiones operativas.

Código

Python, Snowpark, Snowflake ML Registry

Entradas

| **Tipo de Objeto** | **Nombre del Objeto** | **Descripción**                                             |
| ------------------ | --------------------- | ----------------------------------------------------------- |
| Parámetro          | SRC_DATABASE          | Base de datos fuente (desarrollo)                           |
| Parámetro          | TGT_DATABASE          | Base de datos destino (QA/PROD)                             |
| Parámetro          | SRC_MODELS_SCHEMA     | Schema fuente donde reside el Model Registry de desarrollo  |
| Parámetro          | TGT_MODELS_SCHEMA     | Schema destino donde se copiará el modelo                   |
| Parámetro          | MODEL_NAME            | Nombre del modelo particionado a copiar                     |
| Parámetro          | USE_CASE              | Token del caso de uso para tags (ejemplo: CLIENTA_DEFAULT)  |
| Model Registry     | {MODEL_NAME} (SRC)    | Modelo particionado fuente con alias PRODUCTION             |

Salidas

| **Tipo de Objeto** | **Nombre del Objeto**          | **Descripción**                                      |
| ------------------ | ------------------------------ | ---------------------------------------------------- |
| Model Registry     | {MODEL_NAME} (TGT)             | Modelo particionado copiado en ambiente objetivo     |
| Model Tag          | PRODUCTION_{USE_CASE}          | Tag aplicado a la versión activa de producción      |
| Model Tag          | ROLLBACK_VERSION_{USE_CASE}    | Tag aplicado a la versión de rollback                |

Proceso Técnico

- Se inicializa la sesión Snowpark.
- Se conecta al Model Registry fuente y se obtiene el modelo mediante alias PRODUCTION.
- Se verifica si el modelo ya existe en el registry destino:
  - Si NO existe: se copia el modelo completo desde fuente usando registry methods
  - Si SÍ existe: se verifica si la versión específica ya está presente
- Se aplican tags operativos al modelo en el ambiente destino:
  - **PRODUCTION_{USE_CASE}**: identifica la versión activa de producción
  - **ROLLBACK_VERSION_{USE_CASE}**: identifica la versión previa para rollback
- El modelo copiado mantiene todas sus propiedades:
  - Submodelos por segmento encapsulados
  - Lógica de particionamiento por STATS_NTILE_GROUP
  - Métricas y metadatos del modelo
- Los tags permiten gestionar promociones y rollbacks de forma controlada sin depender de alias globales.
- Este script NO copia las tablas de baseline (eso lo hace 07a).

# Inference (Production Data)

### 08_partitioned_inference_batch

Objetivo del Notebook

Ejecutar el proceso de inferencia batch sobre el dataset de inferencia productivo utilizando el modelo particionado identificado mediante tags de producción. Este script detecta automáticamente qué combinaciones (versión del modelo, semana) faltan en la tabla de predicciones y las procesa de forma incremental.

Código

Python, Snowpark, SQL, Snowflake ML Model Registry

Entradas

| **Tipo de Objeto** | **Nombre del Objeto**  | **Descripción**                                                                    |
| ------------------ | ---------------------- | ---------------------------------------------------------------------------------- |
| Parámetro          | Parametros Constantes  | Consultar la sección "Parametros Constantes"                                       |
| Parámetro          | MODEL_NAME             | Nombre del modelo particionado registrado (UNIBOX_CUSTBPR_WEEKLY_FORECAST)        |
| Parámetro          | USE_CASE               | Token del caso de uso para identificar tag de producción (ej: CLIENTA_DEFAULT)    |
| Parámetro          | PRODUCTION_TAG         | Tag que identifica la versión activa (PRODUCTION_{USE_CASE})                      |
| Parámetro          | MIN_INFERENCE_TIME     | Filtro opcional de semanas mínimas a procesar (None = todas)                      |
| Parámetro          | INFERENCE_SAMPLE_FRACTION | Fracción de muestreo opcional (None = dataset completo)                        |
| Tabla              | FEAT_CUSTBPR_WEEKLY__INF | Dataset de inferencia limpio (creado en script 01)                              |
| Tabla              | FEAT_CUSTBPR_WEEKLY__INF_VW | Vista con categorías de clientes para inferencia                            |
| Tabla              | INFERENCE_CUST_CATEGORY_LOOKUP | Lookup de categorías de clientes para inferencia                         |
| Tabla              | GROUND_TRUTH_DATASET_STRUCTURED | Dataset con valores reales para evaluación posterior                    |
| Tabla              | ACTUALS_TABLE_VW       | Vista que expone actuals para join con predicciones                                |
| Modelo             | Partitioned Model (PRODUCTION tag) | Modelo registrado con tag de producción activo                         |

Salidas

| **Tipo de Objeto** | **Nombre del Objeto** | **Descripción**                                              |
| ------------------ | --------------------- | ------------------------------------------------------------ |
| Tabla              | OBS_PREDICTIONS       | Tabla que contiene las predicciones generadas en producción |
| Tabla              | OBS_PREDICTIONS_VW    | Vista transient que join predictions con categorías          |

Proceso Técnico

- Se inicializa la sesión Snowpark.
- Se crean tablas y vistas auxiliares si no existen:
  - INFERENCE_CUST_CATEGORY_LOOKUP: mapeo de categorías de clientes para inferencia
  - FEAT_CUSTBPR_WEEKLY__INF_VW: vista que combina features de inferencia con categorías
  - ACTUALS_TABLE_VW: vista de valores reales desde GROUND_TRUTH_DATASET_STRUCTURED
- Se crea la tabla OBS_PREDICTIONS con esquema completo si no existe.
- Se conecta al Model Registry y se obtiene la versión del modelo mediante el tag:
  - Se lee el tag PRODUCTION_{USE_CASE} del modelo
  - Se extrae el version_name asociado al tag
- Se identifica qué combinaciones (MODEL_VERSION, WEEK) faltan en OBS_PREDICTIONS:
  - Se listan todos los valores únicos de WEEK en el dataset de inferencia
  - Se aplica filtro MIN_INFERENCE_TIME si está configurado
  - Se excluyen combinaciones ya procesadas
- Para cada WEEK faltante:
  - Se carga el batch de datos correspondiente
  - Se aplica muestreo opcional si INFERENCE_SAMPLE_FRACTION está configurado
  - Se ejecuta MODEL()!PREDICT con particionamiento por STATS_NTILE_GROUP
  - Las predicciones se insertan en OBS_PREDICTIONS con metadatos completos:
    - RECORD_ID: hash único por registro
    - MODEL_NAME y MODEL_VERSION (obtenida del tag)
    - ENTITY_MAP: objeto JSON con metadatos del registro (IDs, partition info, data_date)
    - PREDICTION: valor predicho
    - BKCC, CALMONTH, LDTS: metadatos operativos
- Se crea/actualiza la tabla transient OBS_PREDICTIONS_VW mediante join con categorías.
- El proceso es incremental: solo procesa semanas faltantes, permitiendo ejecuciones repetidas sin duplicados.
- Las predicciones almacenadas alimentan los scripts de monitoreo (09b, 09c, 09d).

# ML Observability (Production Data)

Esta etapa monitorea el comportamiento del modelo en producción, detectando drift en datos, predicciones y desempeño mediante comparación con baselines.

### 09a_setup_observability

Objetivo del Notebook

Inicializar la infraestructura de observabilidad para producción, creando las tablas de landing necesarias para almacenar histogramas de drift, métricas de drift con thresholds y alertas, y métricas de desempeño.

Código

Python, Snowpark, SQL

Entradas

| **Tipo de Objeto** | **Nombre del Objeto** | **Descripción**                                           |
| ------------------ | --------------------- | --------------------------------------------------------- |
| Parámetro          | Parametros Constantes | Consultar la sección "Parametros Constantes"              |
| Parámetro          | STORAGE_SCHEMA        | Esquema que contendrá las tablas de monitoreo productivas |
| Parámetro          | N_BINS                | Número de bins para histogramas (default: 20)             |

Salidas

| **Tipo de Objeto** | **Nombre del Objeto**     | **Descripción**                                                            |
| ------------------ | ------------------------- | -------------------------------------------------------------------------- |
| Tabla              | OBS_DATA_HIST             | Tabla para histogramas de features en producción                           |
| Tabla              | OBS_DATA_DRIFT            | Tabla para métricas de data drift con thresholds y alertas                 |
| Tabla              | OBS_PRED_HIST             | Tabla para histogramas de predicciones en producción                       |
| Tabla              | OBS_PRED_DRIFT            | Tabla para métricas de prediction drift con thresholds y alertas           |
| Tabla              | OBS_PERFORMANCE           | Tabla para métricas de desempeño con thresholds y alertas                  |

Proceso Técnico

- Se inicializa la sesión Snowpark.
- Se crean las tablas de landing para histogramas con esquema estandarizado:
  - OBS_DATA_HIST: almacena histogramas de distribución de features en producción
  - OBS_PRED_HIST: almacena histogramas de distribución de predicciones en producción
  - Esquema: RECORD_ID, MODEL_NAME, MODEL_VERSION, ENTITY_MAP, AGGREGATED_COL, AGGREGATED_VALUE, METRIC_COL, METRIC_MAP (OBJECT con histograma), CALMONTH, LDTS
- Se crean las tablas de landing para métricas de drift con esquema estandarizado:
  - OBS_DATA_DRIFT: métricas de divergencia KL para features
  - OBS_PRED_DRIFT: métricas de divergencia KL para predicciones
  - Esquema: RECORD_ID, MODEL_NAME, MODEL_VERSION, ENTITY_MAP, AGGREGATED_COL, AGGREGATED_VALUE, METRIC_COL, METRIC_VALUE, THRESHOLD_WARNING, THRESHOLD_CRITICAL, ALERT_LEVEL, BKCC, CALMONTH, LDTS
- Se crea la tabla de landing para métricas de desempeño:
  - OBS_PERFORMANCE: métricas WAPE, RMSE, MAE, F1_BINARY
  - Esquema: igual que drift tables con thresholds y alert_level
- Estas tablas serán pobladas por los scripts 09b, 09c y 09d.

### 09b_data_drift

Objetivo del Notebook

Detectar cambios en la distribución de las variables de entrada del modelo comparando los datos de inferencia recientes con los histogramas baseline mediante divergencia de Kullback-Leibler (KL).

Código

Python, Snowpark, SQL

Entradas

| **Tipo de Objeto** | **Nombre del Objeto**         | **Descripción**                                                                       |
| ------------------ | ----------------------------- | ------------------------------------------------------------------------------------- |
| Parámetro          | N_BINS                        | Número de bins para histogramas (debe coincidir con baseline)                        |
| Parámetro          | THRESHOLD_WARNING             | Umbral de KL divergence para alerta WARNING (default: 0.1)                            |
| Parámetro          | THRESHOLD_CRITICAL            | Umbral de KL divergence para alerta CRITICAL (default: 0.3)                           |
| Tabla              | FEAT_CUSTBPR_WEEKLY__INF_VW   | Vista con features de inferencia recientes                                            |
| Tabla              | OBS_PREDICTIONS_VW            | Predicciones recientes con metadatos (para identificar versión del modelo)            |
| Tabla              | OBS_DATA_HIST_BL              | Histogramas baseline de features (generados en 06b)                                   |

Salidas

| **Tipo de Objeto** | **Nombre del Objeto** | **Descripción**                                                                       |
| ------------------ | --------------------- | ------------------------------------------------------------------------------------- |
| Tabla              | OBS_DATA_HIST         | Histogramas de features calculados sobre datos de inferencia recientes                |
| Tabla              | OBS_DATA_DRIFT        | Métricas de data drift (KL divergence) con niveles de alerta                          |

Proceso Técnico

- Se inicializa la sesión Snowpark.
- Se identifican combinaciones (MODEL_NAME, MODEL_VERSION) presentes en OBS_PREDICTIONS_VW pero faltantes en OBS_DATA_HIST.
- Para cada combinación faltante:
  - Se consultan las features de inferencia desde FEAT_CUSTBPR_WEEKLY__INF_VW
  - Se identifican las columnas de features (excluyendo metadata: ID_COLS, TIME_COL, TARGET_COL, AGG_COLS, NON_DRIFT_COLS)
  - Para cada feature y cada segmento (STATS_NTILE_GROUP, CUST_CATEGORY):
    - Se calculan histogramas de distribución usando N_BINS bins
    - Se normalizan las frecuencias para obtener distribuciones de probabilidad
    - Se almacenan en OBS_DATA_HIST
  - Se comparan histogramas actuales vs baseline mediante KL divergence:
    - Se hace join entre OBS_DATA_HIST y OBS_DATA_HIST_BL
    - Se calcula KL divergence: sum(p * log(p / q)) donde p=actual, q=baseline
    - Se asigna nivel de alerta según thresholds:
      - OK: KL < THRESHOLD_WARNING
      - WARNING: THRESHOLD_WARNING <= KL < THRESHOLD_CRITICAL
      - CRITICAL: KL >= THRESHOLD_CRITICAL
    - Se almacenan métricas en OBS_DATA_DRIFT
- Las métricas generadas permiten detectar desviaciones en las distribuciones de entrada que pueden afectar el desempeño del modelo.

### 09c_prediction_drift

Objetivo del Notebook

Monitorear cambios en la distribución de las predicciones generadas por el modelo en producción comparando contra histogramas baseline mediante divergencia KL.

Código

Python, Snowpark, SQL

Entradas

| **Tipo de Objeto** | **Nombre del Objeto**    | **Descripción**                                                                    |
| ------------------ | ------------------------ | ---------------------------------------------------------------------------------- |
| Parámetro          | N_BINS                   | Número de bins para histogramas (debe coincidir con baseline)                     |
| Parámetro          | THRESHOLD_WARNING        | Umbral de KL divergence para alerta WARNING (default: 0.1)                         |
| Parámetro          | THRESHOLD_CRITICAL       | Umbral de KL divergence para alerta CRITICAL (default: 0.3)                        |
| Tabla              | OBS_PREDICTIONS_VW       | Vista con predicciones recientes de producción                                     |
| Tabla              | OBS_PRED_HIST_BL         | Histogramas baseline de predicciones (generados en 06c)                            |

Salidas

| **Tipo de Objeto** | **Nombre del Objeto**    | **Descripción**                                                              |
| ------------------ | ------------------------ | ---------------------------------------------------------------------------- |
| Tabla              | OBS_PRED_HIST            | Histogramas de predicciones calculados sobre inferencia reciente             |
| Tabla              | OBS_PRED_DRIFT           | Métricas de prediction drift (KL divergence) con niveles de alerta           |

Proceso Técnico

- Se inicializa la sesión Snowpark.
- Se identifican combinaciones (MODEL_NAME, MODEL_VERSION) presentes en OBS_PREDICTIONS_VW pero faltantes en OBS_PRED_HIST.
- Para cada combinación faltante:
  - Se consultan las predicciones desde OBS_PREDICTIONS_VW
  - Para cada segmento (STATS_NTILE_GROUP, CUST_CATEGORY):
    - Se calculan histogramas de distribución de PREDICTION usando N_BINS bins
    - Se normalizan las frecuencias para obtener distribuciones de probabilidad
    - Se almacenan en OBS_PRED_HIST
  - Se comparan histogramas actuales vs baseline mediante KL divergence:
    - Se hace join entre OBS_PRED_HIST y OBS_PRED_HIST_BL
    - Se calcula KL divergence entre distribuciones
    - Se asigna nivel de alerta según thresholds (OK / WARNING / CRITICAL)
    - Se almacenan métricas en OBS_PRED_DRIFT
- Cambios significativos en la distribución de predicciones pueden indicar:
  - Modificaciones en el comportamiento del modelo
  - Cambios en los patrones de datos de entrada
  - Drift conceptual en el dominio del problema

### 09d_performance_drift

Objetivo del Notebook

Evaluar la degradación del desempeño del modelo en producción comparando métricas actuales contra baseline, detectando deterioro en la capacidad predictiva mediante comparación con thresholds configurados.

Código

Python, Snowpark, SQL

Entradas

| **Tipo de Objeto** | **Nombre del Objeto**   | **Descripción**                                                                            |
| ------------------ | ----------------------- | ------------------------------------------------------------------------------------------ |
| Parámetro          | PERF_JOIN_KEYS          | Claves para join actuals-predictions (CUSTOMER_ID, BRAND_PRES_RET, WEEK)                  |
| Parámetro          | PERF_METRIC_NAMES       | Métricas a calcular (wape, rmse, mae, f1_binary)                                          |
| Parámetro          | THRESHOLD_WARNING       | % de degradación para WARNING (default: 10%)                                               |
| Parámetro          | THRESHOLD_CRITICAL      | % de degradación para CRITICAL (default: 25%)                                              |
| Tabla              | OBS_PREDICTIONS_VW      | Vista con predicciones recientes de producción                                             |
| Tabla              | ACTUALS_TABLE_VW        | Vista con valores reales disponibles para evaluación                                       |
| Tabla              | OBS_PERFORMANCE_BL      | Métricas de desempeño baseline (generadas en 06d)                                          |

Salidas

| **Tipo de Objeto** | **Nombre del Objeto**   | **Descripción**                                                                           |
| ------------------ | ----------------------- | ----------------------------------------------------------------------------------------- |
| Tabla              | OBS_PERFORMANCE         | Métricas de desempeño calculadas sobre producción con niveles de alerta                   |

Proceso Técnico

- Se inicializa la sesión Snowpark.
- Se identifican combinaciones (MODEL_NAME, MODEL_VERSION) presentes en OBS_PREDICTIONS_VW pero faltantes en OBS_PERFORMANCE.
- Para cada combinación faltante:
  - Se hace join entre OBS_PREDICTIONS_VW y ACTUALS_TABLE_VW usando PERF_JOIN_KEYS
  - Para cada segmento (STATS_NTILE_GROUP, CUST_CATEGORY) se calculan métricas:
    - **WAPE**: suma de errores absolutos / suma de valores reales
    - **RMSE**: raíz cuadrada del error cuadrático medio
    - **MAE**: promedio de errores absolutos
    - **F1_BINARY**: métrica binaria (threshold: target > 0)
  - Se comparan métricas actuales vs baseline:
    - Se hace join con OBS_PERFORMANCE_BL
    - Se calcula % de degradación: (actual - baseline) / baseline * 100
    - Para métricas de error (WAPE, RMSE, MAE): degradación = aumento
    - Para métricas de accuracy (F1): degradación = disminución
    - Se asigna nivel de alerta según thresholds:
      - OK: degradación < THRESHOLD_WARNING
      - WARNING: THRESHOLD_WARNING <= degradación < THRESHOLD_CRITICAL
      - CRITICAL: degradación >= THRESHOLD_CRITICAL
  - Se almacenan métricas en OBS_PERFORMANCE con thresholds y alert_level
- El monitoreo continuo permite identificar cuándo el modelo requiere reentrenamiento debido a:
  - Cambios en los datos (concept drift)
  - Cambios en el comportamiento del sistema
  - Deterioro natural del modelo con el tiempo

# Alerting & Notifications

### 10_alertas

Objetivo del Notebook

Este notebook consolida y reporta alertas generadas por la capa de observabilidad, consultando las tablas:

- `OBS_PERFORMANCE` (performance drift)
- `OBS_DATA_DRIFT` (data drift)
- `OBS_PRED_DRIFT` (prediction drift)

El objetivo es identificar registros con `ALERT_LEVEL` en estado **WARNING** o **CRITICAL** (configurable) para un `MODEL_NAME` dado y construir un reporte unificado, listo para ser enviado por correo (HTML) y/o publicado a una cola/event bus (JSON) usando integraciones de notificación de Snowflake.

Código

Python, Snowpark, SQL

Entradas

| **Tipo de Objeto** | **Nombre del Objeto** | **Descripción** |
| --- | --- | --- |
| Parámetro | DATABASE | Base de datos utilizada (BD_AA_DEV) |
| Parámetro | FEATURES_SCHEMA | Esquema donde residen las tablas de observabilidad (SC_FEATURES_BMX) |
| Parámetro | MODEL_NAME | Modelo a monitorear (ej: UNIBOX_CUSTBPR_WEEKLY_FORECAST) |
| Parámetro | REPORT_ALERT_THRESHOLD | Nivel mínimo a reportar (por defecto CRITICAL) |
| Parámetro | LDTS_AFTER | Filtro temporal para considerar solo métricas recientes (LDTS > LDTS_AFTER) |
| Parámetro | PERFORMANCE_METRICS | Métricas a incluir desde performance (ej: rmse, wape) |
| Parámetro | DATA_DRIFT_FEATURE_METRICS | Métricas a incluir desde feature drift (ej: jensen-shannon) |
| Parámetro | DATA_DRIFT_SEGMENT_METRICS | Métricas a incluir desde population drift (ej: population_stability_index) |
| Parámetro | PRED_DRIFT_METRICS | Métricas a incluir desde prediction drift (ej: jensen-shannon) |
| Parámetro | AGGREGATED_COLS | Dimensiones a incluir en el reporte (ej: stats_ntile_group) |
| Tabla | OBS_PERFORMANCE | Métricas de performance con thresholds y `ALERT_LEVEL` |
| Tabla | OBS_DATA_DRIFT | Métricas de data drift con thresholds y `ALERT_LEVEL` |
| Tabla | OBS_PRED_DRIFT | Métricas de prediction drift con thresholds y `ALERT_LEVEL` |
| Configuración | NOTIFICATION_INTEGRATION | Nombre de la integración de email (si se habilita envío) |
| Configuración | QUEUE_NOTIFICATION_INTEGRATION | Nombre de la integración para cola/event bus (si se habilita publicación) |
| Parámetro | EMAIL_RECIPIENTS | Lista de destinatarios del reporte (si se habilita envío) |

Salidas

| **Tipo de Objeto** | **Nombre del Objeto** | **Descripción** |
| --- | --- | --- |
| Reporte (DataFrame) | Consolidated alerts | Resultado consolidado (union) de alertas provenientes de las 3 tablas |
| Reporte (HTML) | email_html | Cuerpo HTML del reporte para envío por email |
| Payload (JSON) | queue_payload | Mensaje JSON para publicar a cola/event bus |
| Notificación (opcional) | SYSTEM$SEND_SNOWFLAKE_NOTIFICATION | Envío por email (TEXT_HTML) y/o publicación (APPLICATION_JSON) si la integración existe |

Proceso Técnico

- Se inicializa la sesión Snowpark y se selecciona `DATABASE` y `FEATURES_SCHEMA`.
- Se define el set de tablas de observabilidad y las métricas relevantes por fuente.
- Se consulta cada tabla filtrando por:
  - `MODEL_NAME`
  - `ALERT_LEVEL >= REPORT_ALERT_THRESHOLD`
  - `METRIC_COL` (solo métricas configuradas por fuente)
  - `AGGREGATED_COL` (solo dimensiones configuradas)
  - `LDTS > LDTS_AFTER` (si aplica)
- Se normaliza el esquema entre fuentes y se agrega la columna `SOURCE_TABLE` para identificar el origen (performance / feature drift / population drift / prediction drift).
- Se construye un reporte consolidado (unión de todas las fuentes) y se ordena por severidad.
- Se genera el HTML del reporte con:
  - Resumen por (MODEL_VERSION, fuente)
  - Secciones de detalle con anclas por fuente y versión
- Se construye un payload JSON con el detalle de alertas para integración con sistemas externos.
- El notebook incluye (comentado) el envío/publicación vía `SYSTEM$SEND_SNOWFLAKE_NOTIFICATION`, que requiere que las Notification Integrations estén configuradas en la cuenta.