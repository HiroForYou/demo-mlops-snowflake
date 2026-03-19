-- ============================================================================
-- CLEANUP SCRIPT V2: Demo MLOps Snowflake (New Naming Convention)
-- ============================================================================
-- Este script elimina todos los recursos creados por los scripts de migración
-- usando la NUEVA convención de nombres propuesta en naming_convention.md
--
-- ADVERTENCIA: Este script eliminará TODOS los recursos creados por los
-- scripts 01-09 con la nueva convención. Asegúrate de hacer un backup si 
-- necesitas conservar algo.
--
-- Uso:
--   1. Revisa las constantes de DATABASE, WAREHOUSE y SCHEMA
--   2. Ejecuta sección por sección o todo junto
--   3. Verifica que no haya errores
-- ============================================================================

-- Configuración inicial
USE ROLE AA_DEV_BMX_PS_DEVELOPER;
USE WAREHOUSE WH_AA_DEV_PS_SO;
USE DATABASE BD_AA_DEV;

-- ============================================================================
-- SECCIÓN 1: MODELOS EN MODEL REGISTRY
-- ============================================================================
-- Elimina los modelos registrados en el Model Registry de Snowflake ML

-- Modelo particionado (creado en 05_create_partitioned_model.py)
-- NUEVO: UNIBOX_CUSTBPR_WEEKLY_FORECAST
DROP MODEL IF EXISTS BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST;

-- Modelos individuales por grupo (creados en 04_many_model_training.py)
-- NUEVO: unibox_custbpr_weekly_forecast__{group}
DROP MODEL IF EXISTS BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST__GROUP_STAT_0_1;
DROP MODEL IF EXISTS BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST__GROUP_STAT_0_2;
DROP MODEL IF EXISTS BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST__GROUP_STAT_0_3;
DROP MODEL IF EXISTS BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST__GROUP_STAT_0_4;
DROP MODEL IF EXISTS BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST__GROUP_STAT_1_1;
DROP MODEL IF EXISTS BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST__GROUP_STAT_1_2;
DROP MODEL IF EXISTS BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST__GROUP_STAT_1_3;
DROP MODEL IF EXISTS BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST__GROUP_STAT_1_4;
DROP MODEL IF EXISTS BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST__GROUP_STAT_2_1;
DROP MODEL IF EXISTS BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST__GROUP_STAT_2_2;
DROP MODEL IF EXISTS BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST__GROUP_STAT_2_3;
DROP MODEL IF EXISTS BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST__GROUP_STAT_2_4;
DROP MODEL IF EXISTS BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST__GROUP_STAT_3_1;
DROP MODEL IF EXISTS BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST__GROUP_STAT_3_2;
DROP MODEL IF EXISTS BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST__GROUP_STAT_3_3;
DROP MODEL IF EXISTS BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST__GROUP_STAT_3_4;

-- ============================================================================
-- SECCIÓN 2: STAGES Y TAGS
-- ============================================================================
-- Elimina stages y tags usados en el proceso

-- Stage para Many Model Training (creado en 04_many_model_training.py)
DROP STAGE IF EXISTS BD_AA_DEV.SC_MODELS_BMX.MMT_MODELS;

-- Tag para versionado de modelos candidatos (creado en 07a/07b)
DROP TAG IF EXISTS BD_AA_DEV.SC_STORAGE_BMX_PS.CANDIDATE_VERSION;
DROP TAG IF EXISTS BD_AA_DEV.SC_MODELS_BMX.CANDIDATE_VERSION;

-- ============================================================================
-- SECCIÓN 3: TABLAS DE OBSERVABILIDAD (PRODUCCIÓN)
-- ============================================================================
-- Elimina tablas de drift y performance en el schema de producción
-- (creadas en 09a_setup_observability.py)
-- NUEVO: OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__*

USE SCHEMA SC_FEATURES_BMX;

DROP TABLE IF EXISTS OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__DATA_HIST;
DROP TABLE IF EXISTS OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__DATA_DRIFT;
DROP TABLE IF EXISTS OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PRED_HIST;
DROP TABLE IF EXISTS OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PRED_DRIFT;
DROP TABLE IF EXISTS OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PERF;

-- Tablas de predicciones en producción (creadas en 08_partitioned_inference_batch.py)
DROP TABLE IF EXISTS OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PRED;
DROP TABLE IF EXISTS OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PRED_VW;

-- Tablas auxiliares de producción
DROP TABLE IF EXISTS INFERENCE_DATASET_CLEANED;
DROP TABLE IF EXISTS INFERENCE_CUST_CATEGORY_LOOKUP;
DROP TABLE IF EXISTS GROUND_TRUTH_DATASET_STRUCTURED;
DROP VIEW IF EXISTS FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST__INF_VW;
DROP VIEW IF EXISTS ACTUALS_TABLE_VW;

-- ============================================================================
-- SECCIÓN 4: TABLAS DE BASELINES (DESARROLLO)
-- ============================================================================
-- Elimina tablas de baselines en el schema de desarrollo
-- (creadas en 06a_setup_baselines.py)
-- NUEVO: OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__*_BL

USE SCHEMA SC_STORAGE_BMX_PS;

-- Tablas de drift baselines
DROP TABLE IF EXISTS OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__DATA_HIST_BL;
DROP TABLE IF EXISTS OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PRED_HIST_BL;
DROP TABLE IF EXISTS OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PERF_BL;

-- Tablas de predicciones baseline
DROP TABLE IF EXISTS OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PRED_BL;
DROP TABLE IF EXISTS OBS_UNIBOX_CUSTBPR_WEEKLY_FORECAST__PRED_BL_VW;

-- Tablas auxiliares de baseline
DROP TABLE IF EXISTS TRAIN_CUST_CATEGORY_LOOKUP;
DROP VIEW IF EXISTS FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST__HOLDOUT_VW;

-- ============================================================================
-- SECCIÓN 5: TABLAS DE HIPERPARÁMETROS Y EXPERIMENTOS
-- ============================================================================
-- Elimina tabla de resultados de HPO (creada en 03_hyperparameter_search.py)

DROP TABLE IF EXISTS BD_AA_DEV.SC_MODELS_BMX.HPO_UNIBOX_CUSTBPR_WEEKLY_FORECAST;

-- NOTA: Los experimentos de ML (ML Experiments) no se pueden eliminar con DROP
-- Para limpiar experimentos, usa:
--   1. Snowsight UI: Data > ML Functions > Experiments
--   2. O el comando: DROP EXPERIMENT <experiment_name>
-- Experimentos creados (NUEVOS NOMBRES):
--   - EXP_UNIBOX_CUSTBPR_WEEKLY_FORECAST_RANDOM_YYYYMMDD (script 03)
--   - EXP_UNIBOX_CUSTBPR_WEEKLY_FORECAST_BAYESIAN_YYYYMMDD (script 03b)

-- ============================================================================
-- SECCIÓN 6: TABLAS DE FEATURES Y DATOS LIMPIOS
-- ============================================================================
-- Elimina tablas de features y datos procesados

USE SCHEMA SC_FEATURES_BMX;

-- Tabla de features (creada en 02_feature_store_setup.py)
-- NUEVO: FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST
DROP TABLE IF EXISTS FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST;

-- Vistas del feature store
DROP VIEW IF EXISTS FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST__TRAIN_VW;
DROP VIEW IF EXISTS FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST__INF_VW;
DROP VIEW IF EXISTS FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST__HOLDOUT_VW;

-- Tablas de datos limpios (creadas en 01_data_validation_and_cleaning.py)
-- NUEVO: Ahora en SC_FEATURES_BMX con prefijo FEAT_
DROP TABLE IF EXISTS FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST__TRAIN;
DROP TABLE IF EXISTS FEAT_UNIBOX_CUSTBPR_WEEKLY_FORECAST__HOLDOUT;

USE SCHEMA SC_STORAGE_BMX_PS;

-- Tabla de inference cleaned (sigue en SC_STORAGE_BMX_PS)
DROP TABLE IF EXISTS INFERENCE_DATASET_CLEANED;

-- Tabla temporal de thresholds (creada temporalmente en 01)
-- DROP TABLE IF EXISTS TRAIN_DATASET_CLEANED_TEMP_THRESHOLDS; -- Es TEMPORARY, se elimina automáticamente

-- ============================================================================
-- SECCIÓN 7: TABLAS FUENTE (OPCIONAL)
-- ============================================================================
-- ADVERTENCIA: Estas son las tablas originales de datos estructurados.
-- Solo descomentar si quieres eliminar TODO y empezar desde cero.

-- DROP TABLE IF EXISTS BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_STRUCTURED;
-- DROP TABLE IF EXISTS BD_AA_DEV.SC_STORAGE_BMX_PS.INFERENCE_DATASET_STRUCTURED;
-- DROP TABLE IF EXISTS BD_AA_DEV.SC_STORAGE_BMX_PS.GROUND_TRUTH_DATASET_STRUCTURED;

-- ============================================================================
-- SECCIÓN 8: VERIFICACIÓN
-- ============================================================================
-- Verifica que todos los recursos se eliminaron correctamente

-- Modelos restantes
SHOW MODELS IN SCHEMA BD_AA_DEV.SC_MODELS_BMX;

-- Stages restantes en SC_MODELS_BMX
SHOW STAGES IN SCHEMA BD_AA_DEV.SC_MODELS_BMX;

-- Tags restantes
SHOW TAGS IN SCHEMA BD_AA_DEV.SC_STORAGE_BMX_PS;
SHOW TAGS IN SCHEMA BD_AA_DEV.SC_MODELS_BMX;

-- Tablas restantes en SC_STORAGE_BMX_PS
SHOW TABLES IN SCHEMA BD_AA_DEV.SC_STORAGE_BMX_PS;

-- Tablas restantes en SC_FEATURES_BMX
SHOW TABLES IN SCHEMA BD_AA_DEV.SC_FEATURES_BMX;

-- Views restantes
SHOW VIEWS IN SCHEMA BD_AA_DEV.SC_STORAGE_BMX_PS;
SHOW VIEWS IN SCHEMA BD_AA_DEV.SC_FEATURES_BMX;

-- ============================================================================
-- FIN DEL SCRIPT DE LIMPIEZA V2
-- ============================================================================

-- NOTA FINAL:
-- Si después de ejecutar este script aún ves objetos residuales, puedes:
-- 1. Ejecutar los comandos SHOW de la sección 8 para identificarlos
-- 2. Eliminarlos manualmente con DROP
-- 3. Si hay problemas de dependencias, usa CASCADE:
--    DROP TABLE <nombre> CASCADE;
--
-- Para volver a la convención ANTIGUA, usa: cleanup_all_resources.sql
