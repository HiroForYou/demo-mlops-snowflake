# Diccionario de Datos: Pipeline de MLOps (Scripts de Migración)

Este documento detalla todas las tablas generadas o utilizadas en los scripts de migración (01_data_validation_and_cleaning.py hasta 09d_performance_drift.py), agrupadas por familia funcional y siguiendo la estructura solicitada.

## 1. Tablas de Monitoreo: Métricas de Drift
**Tablas aplicables:** `OBS_DATA_DRIFT`, `OBS_PRED_DRIFT`
**Descripción:** Almacenan las métricas de drift calculadas, junto con sus umbrales y niveles de alerta correspondientes.

| Columna | Tipo | PK/FK | Nullable | Descripción | Ejemplo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RECORD_ID | string | PK | No | ID único del registro, hash determinístico de los campos clave de la evaluación | f25ec723ef... |
| MODEL_NAME | string | FK | No | Nombre del modelo evaluado | UNIBOX_CUSTBPR_WEEKLY_FORECAST |
| MODEL_VERSION | string | | No | Versión específica del modelo (generada dinámicamente, ej. v_YYYYMMDD_HHMM) | v_20260301_0900 |
| ENTITY_MAP | map<string,string> | | Yes | Mapa con dimensiones de la evaluación (e.g., feature_name, week, data_date) | `{"feature_name":"wines_liquor", "week":"202602", "data_date":"2026-01-05"}` |
| AGGREGATED_COL | string | | No | Nombre de la columna usada para agrupar o segmentar la evaluación | STATS_NTILE_GROUP |
| AGGREGATED_VALUE | string | | No | Valor numérico o categórico del segmento evaluado | 3 |
| METRIC_COL | string | | No | Nombre de la métrica de drift o feature evaluado | psi |
| METRIC_VALUE | float | | Yes | Valor numérico resultante de la métrica de drift calculada | 0.045 |
| WARNING_THRESHOLD | float | | No | Umbral configurado para nivel warning | 0.1 |
| CRITICAL_THRESHOLD | float | | No | Umbral configurado para nivel critical | 0.2 |
| ALERT_LEVEL | integer | | No | Nivel de alerta calculado: 0=ok, 1=warning, 2=critical | 0 |
| BKCC | string | | No | Código de unidad de negocio del entorno | MXBEB |
| CALMONTH | string | | Yes | Mes calendario en formato YYYYMM, usado para particionamiento | 202501 |
| LDTS | timestamp | | No | Load datetime stamp, momento de carga y evaluación del registro | 2025-01-12T10:52:31.800+00:00 |

## 2. Tablas de Monitoreo: Performance
**Tablas aplicables:** `OBS_PERFORMANCE`, `OBS_PERFORMANCE_BL`
**Descripción:** Registran las métricas de desempeño del modelo, comparando contra actuals y evaluando la desviación contra la línea base (baseline).

| Columna | Tipo | PK/FK | Nullable | Descripción | Ejemplo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RECORD_ID | string | PK | No | ID único del cálculo | d80c110d8d... |
| MODEL_NAME | string | FK | No | Nombre del modelo predictivo evaluado | UNIBOX_CUSTBPR_WEEKLY_FORECAST |
| MODEL_VERSION | string | | No | Versión operante del modelo (generada dinámicamente, ej. v_YYYYMMDD_HHMM) | v_20260301_0900 |
| ENTITY_MAP | map<string,string> | | Yes | Mapa dimensional del registro evaluado (e.g., week, data_date) | `{"week":"202602", "data_date":"2026-01-05"}` |
| AGGREGATED_COL | string | | No | Columna de agrupación para segmentación de desempeño | STATS_NTILE_GROUP |
| AGGREGATED_VALUE | string | | No | Valor del grupo que ha sido evaluado | 3 |
| METRIC_COL | string | | No | Nombre de la métrica de desempeño utilizada | rmse |
| METRIC_VALUE | float | | Yes | Valor numérico del desempeño | 125.4 |
| METRIC_DRIFT | float | | Yes | Diferencia entre la métrica actual vs baseline (solo en OBS_PERFORMANCE) | 5.2 |
| WARNING_THRESHOLD | float | | No | Umbral configurado para advertir caída de desempeño rel. al baseline | 10.0 |
| CRITICAL_THRESHOLD | float | | No | Umbral crítico de caída de desempeño rel. al baseline | 20.0 |
| ALERT_LEVEL | integer | | No | Nivel de alerta calculado (0=ok, 1=warn, 2=crit) | 0 |
| BKCC | string | | No | Código de unidad de negocio | MXBEB |
| CALMONTH | string | | Yes | Mes calendario evaluado | 202501 |
| LDTS | timestamp | | No | Fecha de almacenamiento | 2025-01-12T10:52:31.800+00:00 |

## 3. Tablas de Monitoreo: Histogramas
**Tablas aplicables:** `OBS_DATA_HIST`, `OBS_PRED_HIST`, `OBS_DATA_HIST_BL`, `OBS_PRED_HIST_BL`
**Descripción:** Almacenan de forma compactada los bins o deciles y sus respectivas frecuencias de distribución temporal para features y predicciones (usados para los gráficos de la vista de monitor).

| Columna | Tipo | PK/FK | Nullable | Descripción | Ejemplo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RECORD_ID | string | PK | No | ID único de la distribución calculada | ab12cd34... |
| MODEL_NAME | string | FK | No | Nombre del modelo | UNIBOX_CUSTBPR_WEEKLY_FORECAST |
| MODEL_VERSION | string | | No | Versión del modelo evaluado (generada dinámicamente, ej. v_YYYYMMDD_HHMM) | v_20260301_0900 |
| ENTITY_MAP | map<string,string> | | Yes | Diccionario de dimensiones de la distribución (e.g., feature_name, week) | `{"feature_name":"wines_liquor", "week":"202602", "data_date":"2026-01-05"}` |
| AGGREGATED_COL | string | | No | Columna usada para separar y agrupar la evaluación | STATS_NTILE_GROUP |
| AGGREGATED_VALUE | string | | Yes | Valor del segmento estudiado | 3 |
| METRIC_COL | string | | No | Nombre de la característica o output que fue tabulado | sum_past_12_weeks |
| METRIC_MAP | object | | Yes | Objeto JSON con los bordes de contenedores y recuentos estadísticos | `{"bin_edges": [0,10,20], "hist": [50,30]}` |
| CALMONTH | string | | Yes | Mes abarcado por el histograma | 202501 |
| LDTS | timestamp | | No | Timestamp de la creación en la base de datos | 2025-01-12T10:52:31.800+00:00 |

## 4. Tablas de Predicciones y Generación de Baselines
**Tablas aplicables:** `OBS_PREDICTIONS`, `OBS_PREDICTIONS_BL`
**Descripción:** Tablas planas temporales o landing que contienen las predicciones a nivel de registro que son emitidas por el modelo en producción, así como su inferencia respectiva proyectada sobre datos históricos (Baseline).

| Columna | Tipo | PK/FK | Nullable | Descripción | Ejemplo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RECORD_ID | string | PK | No | UUID del registro único de la nueva predicción | 1a2b3c... |
| MODEL_NAME | string | FK | No | Nombre del modelo predictivo | UNIBOX_CUSTBPR_WEEKLY_FORECAST |
| MODEL_VERSION | string | | No | Identificador de la versión con que fue generada (e.g., v_YYYYMMDD_HHMM) | v_20260301_0900 |
| ENTITY_MAP | object | | Yes | JSON con claves iterables de inferencia (customer_id, week, partition_value, etc) | `{"customer_id":"100456", "week":"202602", "partition_value":"3"}` |
| PREDICTION | float | | Yes | Valor continuo estimado o inferido estadísticamente por el modelo | 340.5 |
| BKCC | string | | No | Código de unidad del sector de negocio de despliegue | MXBEB |
| CALMONTH | string | | Yes | Mes calendario del output inferido | 202501 |
| LDTS | timestamp | | No | Fecha en la que la inferencia fue finalmente consolidada | 2025-01-12T10:52:31.800+00:00 |

## 5. Tablas de Base y Feature Store (Preparación de datos)
**Tablas aplicables:** `FEAT_CUSTBPR_WEEKLY__TRAIN`, `FEAT_CUSTBPR_WEEKLY__HOLDOUT`, `FEAT_CUSTBPR_WEEKLY__INF`, `FEAT_CUSTBPR_WEEKLY`
**Descripción:** Tablas originadas en etapa de Data Validation o materializadas en la capa de Features (archivos 01_ y 02_). Contienen el universo de features para conformar las muestras segmentadas o la inferencia productiva.

| Columna | Tipo | PK/FK | Nullable | Descripción | Ejemplo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CUSTOMER_ID | string | PK | No | Identificador del cliente / punto de venta en SnowFlake | 100456 |
| BRAND_PRES_RET | string | PK | No | Identificador asociado a la categoría de producto/marca/contenedor | COCA-COLA 600ML |
| PROD_KEY | string | | Yes | Clave unificadora secundaria provista | P_123 |
| WEEK | integer/str | PK | No | Semana calendario donde ocurre o cerró el evento transaccional | 202501 |
| STATS_NTILE_GROUP | integer | | Yes | Llave n-tile derivada para repartir la carga del modelo eficientemente | 5 |
| UNI_BOX_WEEK | float | | Yes | Etiqueta de la caja a predecir (Target). *Nulo en instancias the Inferencia.* | 15.2 |
| SUM_PAST_12_WEEKS | float | | Yes | **(Feature)** Ventas agrupadas registradas a 12 semanas | 150.5 |
| WINES_LIQUOR | integer | | Yes | **(Feature Booleano)** Categoría dummy vinculada al tipo de consumo | 1 |

*(Nota: De igual forma al feature mostrado en el esquema, existe iterativamente docenas más en la columna `MAX_`, `MIN_`, `AVG_`, y `NUM_` que mantienen idéntica estructura y nulanidad).*

## 6. Vistas y Tablas de Referencia Transitorias (Lookups)
**Tablas aplicables:** `TRAIN_CUST_CATEGORY_LOOKUP`, `INFERENCE_CUST_CATEGORY_LOOKUP`, `OBS_PREDICTIONS_VW`, `ACTUALS_TABLE_VW`
**Descripción:** Son tablas relacionales maestras ("lookup tables") usadas para expandir los features originarios con clasificaciones secundarias como `cust_category`, o materializar vistas simplificadas de consultas robustas para optimizar procesamiento posterior.

| Columna | Tipo | PK/FK | Nullable | Descripción | Ejemplo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CUSTOMER_ID | string | FK | No | ID original del cliente que permite correlación bidireccional | 100456 |
| BRAND_PRES_RET | string | FK | No | ID primario representativo de las marcas a englobar | COCA-COLA 600ML |
| WEEK | integer | FK | No | Rango temporal semanal de evaluación referenciada | 202501 |
| STATS_NTILE_GROUP | integer | | Yes | El agrupador de distribución proveniente del Feature original | 3 |
| CUST_CATEGORY | string | | Yes | Dimensión clasificada jerárquica derivada para segmentaciones finales | groceries |

## 7. Tablas de Tuning y Entrenamiento (Hiperparámetros)
**Tablas aplicables:** `HPO_{MODEL_NAME}` (ej. `HPO_UNIBOX_CUSTBPR_WEEKLY_FORECAST`)
**Descripción:** Creada en `03_hyperparameter_search.py` y `03b_`. Almacena los resultados locales de la búsqueda de hiperparámetros (Random Search o BayesOpt) por cada partición iterada. Se usa como *fallback* o repositorio si Snowflake ML Experiments no está activo.

| Columna | Tipo | PK/FK | Nullable | Descripción | Ejemplo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| search_id | string | PK | No | Identificador único de la búsqueda hiperparamétrica | bayes_group_stat_3_4_20260301_090000 |
| group_name | string | | No | Nombre de la partición segmentada sobre la cual se entrena | group_stat_3_4 |
| algorithm | string | | No | Modelo algorítmico utilizado | XGBRegressor |
| best_params | variant | | No | JSON Variant con diccionario exacto de hiperparámetros | `{"n_estimators": 100, "max_depth": 5}` |
| best_cv_rmse | float | | Yes | RMSE final estimado según trials del Tuner | 1.45 |
| val_rmse | float | | Yes | RMSE real evaluado sobre el conjunto temporal de validación estructurado | 1.82 |
| val_mae | float | | Yes | MAE evaluado localmente sobre el conjunto temporal de validación | 0.95 |
| n_iter | integer | | Yes | Cantidad de iteraciones o trials configurados globalmente | 15 |
| sample_size | integer | | Yes | Cantidad final de registros usados para entrenar dicha partición | 50000 |
| created_at | timestamp | | Yes | Timestamp default de la creación de la fila (registro log) | 2026-03-01T09:35:00.000+00:00 |

*(Nota: Comparten la misma estructura material básica en llaves que la tabla generada `FEAT_CUSTBPR_WEEKLY__TRAIN` pero sin la limitación de registros anómalos o limpieza minuciosa, sirviendo a `CUSTOMER_ID`, `BRAND_PRES_RET`, `PROD_KEY` y `WEEK` de llaves maestras).*
