# Resumen de Cambios: Feature Store Compartido y Observabilidad Genérica

**Fecha:** 17 de marzo de 2026  
**Alcance:** Actualización completa de nomenclatura en 16 scripts Python y documentación

---

## 🎯 Cambios Principales

### 1. Feature Store: De 1:1 con Modelo a Compartido por Entidad

**Antes (v2):**
```python
FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST         # 1:1 con el modelo
FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST__TRAIN
FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST__HOLDOUT
FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST__INF
```

**Ahora (v3):**
```python
FEAT_CUSTBPR_WEEKLY                          # Compartido por entidad + frecuencia
FEAT_CUSTBPR_WEEKLY__TRAIN
FEAT_CUSTBPR_WEEKLY__HOLDOUT
FEAT_CUSTBPR_WEEKLY__INF
FEAT_CUSTBPR_WEEKLY__TRAIN_VW                # Vistas (opcional)
FEAT_CUSTBPR_WEEKLY__HOLDOUT_VW              # Vista usada en baselines
FEAT_CUSTBPR_WEEKLY__INF_VW                  # Vista usada en inferencia
```

**Patrón:** `FEAT_<entity>_<frequency>`

**Beneficios:**
- ✅ Múltiples modelos pueden compartir las mismas features
- ✅ Reduce duplicación de datos
- ✅ Facilita experimentación con nuevos modelos
- ✅ Mejora mantenibilidad (una fuente de verdad)
- ✅ Permite evolución independiente de features y modelos

---

### 2. Observabilidad: De Por-Modelo a Tablas Genéricas

**Antes (v2):**
```python
OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PRED
OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__DATA_DRIFT
OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__DATA_HIST
OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__DATA_HIST_BL
OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PRED_DRIFT
OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PRED_HIST
OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PRED_HIST_BL
OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PERF
OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PERF_BL
```

**Ahora (v3):**
```python
OBS_PREDICTIONS                              # Genéricas (sin nombre de modelo)
OBS_DATA_DRIFT
OBS_DATA_HIST
OBS_DATA_HIST_BL
OBS_PRED_DRIFT
OBS_PRED_HIST
OBS_PRED_HIST_BL
OBS_PERFORMANCE
OBS_PERFORMANCE_BL
```

**Patrón:** `OBS_<metric_type>`

**Identificación de Modelo:**
Cada registro incluye columnas estándar:
```sql
MODEL_NAME       VARCHAR(64)   -- Nombre del modelo (e.g., 'UNIBOX_CUSTBPR_WEEKLY_FORECAST')
MODEL_VERSION    VARCHAR(32)   -- Versión del modelo (e.g., 'v_20260317_1430')
```

**Beneficios:**
- ✅ Facilita comparación entre modelos
- ✅ Reduce proliferación de tablas (1 tabla para N modelos)
- ✅ Simplifica queries y dashboards (una sola fuente)
- ✅ Mejora escalabilidad (agregar modelo = 0 nuevas tablas)
- ✅ Permite análisis cross-model

---

## 📋 Scripts Actualizados

Todos los scripts siguientes fueron actualizados con las nuevas convenciones:

### Scripts de Pipeline (16 scripts)

| Script | Cambios Aplicados |
|--------|-------------------|
| `01_data_validation_and_cleaning.py` | Feature store compartido |
| `02_feature_store_setup.py` | Feature store compartido |
| `03_hyperparameter_search.py` | Feature store compartido |
| `03b_hyperparameter_search_bayesian.py` | Feature store compartido |
| `04_many_model_training.py` | Feature store compartido |
| `05_create_partitioned_model.py` | Feature store compartido |
| `06a_setup_baselines.py` | Feature store + OBS genérica |
| `06b_data_drift_baseline.py` | Feature store + OBS genérica |
| `06c_prediction_drift_baseline.py` | OBS genérica |
| `06d_performance_drift_baseline.py` | Feature store + OBS genérica |
| `07_environment_change.py` | OBS genérica |
| `08_partitioned_inference_batch.py` | Feature store + OBS genérica |
| `09a_setup_observability.py` | Feature store + OBS genérica |
| `09b_data_drift.py` | Feature store + OBS genérica |
| `09c_prediction_drift.py` | OBS genérica |
| `09d_performance_drift.py` | Feature store + OBS genérica |

---

## 📚 Documentación Actualizada

### 1. `migration/docs/naming_convention.md`

**Sección 2: Feature Store Naming**
- Actualizada estrategia de 1:1 a "Shared Feature Store (Entity-Based)"
- Nuevo patrón: `FEAT_<entity>_<frequency>`
- Tabla de migración: antiguo → nuevo
- Documentados beneficios del feature store compartido

**Sección 4: Observability Tables**
- Actualizada estrategia de por-modelo a "Generic Observability (Model-Agnostic)"
- Nuevo patrón: `OBS_<metric_type>`
- Tabla de migración: antiguo → nuevo
- Documentadas columnas estándar de identificación de modelo
- Documentados beneficios de tablas genéricas

### 2. Scripts de Limpieza SQL

Creados 3 scripts para distintas versiones:

| Script | Descripción |
|--------|-------------|
| `cleanup_all_resources.sql` (v1) | Convención antigua (nombres originales de Databricks) |
| `cleanup_all_resources_v2.sql` (v2) | Feature store 1:1 con modelo + OBS por modelo |
| `cleanup_all_resources_v3.sql` (v3) | **Feature store compartido + OBS genérica** ← ACTUAL |

---

## 🔧 Patrón de Implementación

Todos los scripts Python ahora siguen este patrón consistente:

```python
# Model name (base for all model-specific objects)
MODEL_NAME = "UNIBOX_CUSTBPR_WEEKLY_FORECAST"

# Feature store name (shared by entity and frequency, not tied to model)
FEATURE_STORE_NAME = "FEAT_CUSTBPR_WEEKLY"

# Feature store tables
TRAIN_TABLE = f"{DATABASE}.{FEATURES_SCHEMA}.{FEATURE_STORE_NAME}__TRAIN"
HOLDOUT_TABLE = f"{DATABASE}.{FEATURES_SCHEMA}.{FEATURE_STORE_NAME}__HOLDOUT"
INFERENCE_TABLE = f"{DATABASE}.{FEATURES_SCHEMA}.{FEATURE_STORE_NAME}__INF"

# Feature store views (used in some scripts for additional transformations)
TRAIN_VW = f"{DATABASE}.{FEATURES_SCHEMA}.{FEATURE_STORE_NAME}__TRAIN_VW"
HOLDOUT_VW = f"{DATABASE}.{FEATURES_SCHEMA}.{FEATURE_STORE_NAME}__HOLDOUT_VW"
INFERENCE_VW = f"{DATABASE}.{FEATURES_SCHEMA}.{FEATURE_STORE_NAME}__INF_VW"

# Observability tables (generic, shared across models)
PREDICTIONS_TABLE = "OBS_PREDICTIONS"
PREDICTIONS_VW = "OBS_PREDICTIONS_VW"
DATA_DRIFT_TABLE = "OBS_DATA_DRIFT"
PRED_DRIFT_TABLE = "OBS_PRED_DRIFT"
PERFORMANCE_TABLE = "OBS_PERFORMANCE"
PREDICTIONS_BL = "OBS_PREDICTIONS_BL"
PREDICTIONS_BL_VW = "OBS_PREDICTIONS_BL_VW"
```

---

## 🚀 Próximos Pasos

1. **Testing:**
   - Ejecutar todos los scripts 01-09 con la nueva nomenclatura
   - Verificar que las tablas se crean correctamente
   - Validar que queries de observabilidad filtran correctamente por `MODEL_NAME`

2. **Migración de Datos Existentes (si aplica):**
   - Si hay datos en tablas antiguas, crear scripts de migración
   - Considerar usar `INSERT INTO ... SELECT` para copiar datos
   - Agregar columnas `MODEL_NAME` y `MODEL_VERSION` a registros existentes

3. **Actualización de Dashboards:**
   - Ajustar queries de Snowsight/Tableau para usar nuevas tablas
   - Agregar filtros por `MODEL_NAME` en visualizaciones
   - Aprovechar capacidad de comparar múltiples modelos

4. **Documentación de Equipo:**
   - Comunicar cambios al equipo de Data Science
   - Actualizar wikis/READMEs internos
   - Crear ejemplos de queries cross-model

---

## 📊 Tabla de Migración Completa

### Feature Store

| Antes (v2) | Ahora (v3) |
|-----------|-----------|
| `FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST` | `FEAT_CUSTBPR_WEEKLY` |
| `FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST__TRAIN` | `FEAT_CUSTBPR_WEEKLY__TRAIN` |
| `FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST__HOLDOUT` | `FEAT_CUSTBPR_WEEKLY__HOLDOUT` |
| `FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST__INF` | `FEAT_CUSTBPR_WEEKLY__INF` |
| `FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST__TRAIN_VW` | `FEAT_CUSTBPR_WEEKLY__TRAIN_VW` |
| `FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST__HOLDOUT_VW` | `FEAT_CUSTBPR_WEEKLY__HOLDOUT_VW` |
| `FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST__INF_VW` | `FEAT_CUSTBPR_WEEKLY__INF_VW` |

### Tablas de Observabilidad

| Antes (v2) | Ahora (v3) |
|-----------|-----------|
| `OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PRED` | `OBS_PREDICTIONS` |
| `OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PRED_VW` | `OBS_PREDICTIONS_VW` |
| `OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__DATA_DRIFT` | `OBS_DATA_DRIFT` |
| `OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__DATA_HIST` | `OBS_DATA_HIST` |
| `OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__DATA_HIST_BL` | `OBS_DATA_HIST_BL` |
| `OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PRED_DRIFT` | `OBS_PRED_DRIFT` |
| `OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PRED_HIST` | `OBS_PRED_HIST` |
| `OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PRED_HIST_BL` | `OBS_PRED_HIST_BL` |
| `OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PERF` | `OBS_PERFORMANCE` |
| `OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PERF_BL` | `OBS_PERFORMANCE_BL` |
| `OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PRED_BL` | `OBS_PREDICTIONS_BL` |
| `OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PRED_BL_VW` | `OBS_PREDICTIONS_BL_VW` |

---

## ✅ Checklist de Completitud

- [x] Actualizar `naming_convention.md` (Feature Store)
- [x] Actualizar `naming_convention.md` (Observabilidad)
- [x] Actualizar script 01 (data validation)
- [x] Actualizar script 02 (feature store)
- [x] Actualizar script 03 (HPO random)
- [x] Actualizar script 03b (HPO bayesian)
- [x] Actualizar script 04 (many model training)
- [x] Actualizar script 05 (partitioned model)
- [x] Actualizar script 06a (setup baselines)
- [x] Actualizar script 06b (data drift baseline)
- [x] Actualizar script 06c (prediction drift baseline)
- [x] Actualizar script 06d (performance drift baseline)
- [x] Actualizar script 07 (environment change)
- [x] Actualizar script 08 (inference batch)
- [x] Actualizar script 09a (setup observability)
- [x] Actualizar script 09b (data drift)
- [x] Actualizar script 09c (prediction drift)
- [x] Actualizar script 09d (performance drift)
- [x] Crear `cleanup_all_resources_v3.sql`
- [x] Crear documento de resumen de cambios

---

## 🎓 Ejemplo de Query Cross-Model

Con la nueva estructura genérica, ahora puedes comparar múltiples modelos fácilmente:

```sql
-- Comparar performance de múltiples modelos
SELECT 
    MODEL_NAME,
    MODEL_VERSION,
    ENTITY_MAP:STATS_NTILE_GROUP::STRING AS SEGMENT,
    METRIC_COL,
    METRIC_VALUE,
    METRIC_DRIFT,
    ALERT_LEVEL
FROM OBS_PERFORMANCE
WHERE ENTITY_MAP:week::STRING = '2026-03-17'
  AND MODEL_NAME IN (
    'UNIBOX_CUSTBPR_WEEKLY_FORECAST',
    'PROB_CUSTBPR_WEEKLY_CLASSIF'
  )
ORDER BY MODEL_NAME, SEGMENT, METRIC_COL;
```

---

**Fin del documento de resumen**
