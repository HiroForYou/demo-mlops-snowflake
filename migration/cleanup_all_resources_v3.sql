-- ============================================================================
-- CLEANUP SCRIPT V3: Demo MLOps Snowflake (Shared Feature Store & Generic OBS)
-- ============================================================================
-- Este script elimina todos los recursos creados por los scripts de migración
-- usando la convención de nombres ACTUALIZADA con:
--   - Feature Store compartido (por entidad + frecuencia, sin nombre de modelo)
--   - Tablas de observabilidad genéricas (sin nombre de modelo)
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
DROP MODEL IF EXISTS BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST;

-- Modelos individuales por grupo (creados en 04_many_model_training.py)
-- Nomenclatura: {model_name}__{group}
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
-- SECCIÓN 3: TABLAS DE OBSERVABILIDAD (PRODUCCIÓN) - GENÉRICAS
-- ============================================================================
-- Elimina tablas de drift y performance en el schema de producción
-- (creadas en 09a_setup_observability.py)
-- NUEVO: Tablas genéricas compartidas entre modelos (sin nombre de modelo)

USE SCHEMA SC_FEATURES_BMX;

-- Tablas de drift de datos (data drift)
DROP TABLE IF EXISTS OBS_DATA_HIST;
DROP TABLE IF EXISTS OBS_DATA_DRIFT;

-- Tablas de drift de predicciones (prediction drift)
DROP TABLE IF EXISTS OBS_PRED_HIST;
DROP TABLE IF EXISTS OBS_PRED_DRIFT;

-- Tablas de performance
DROP TABLE IF EXISTS OBS_PERFORMANCE;

-- Tablas de predicciones en producción (creadas en 08_partitioned_inference_batch.py)
DROP TABLE IF EXISTS OBS_PREDICTIONS;
DROP VIEW IF EXISTS OBS_PREDICTIONS_VW;

-- Tablas auxiliares de producción
DROP TABLE IF EXISTS INFERENCE_CUST_CATEGORY_LOOKUP;
DROP TABLE IF EXISTS GROUND_TRUTH_DATASET_STRUCTURED;
DROP VIEW IF EXISTS FEAT_CUSTBPR_WEEKLY__INF_VW;
DROP VIEW IF EXISTS ACTUALS_TABLE_VW;

-- ============================================================================
-- SECCIÓN 4: TABLAS DE BASELINES (DESARROLLO) - GENÉRICAS
-- ============================================================================
-- Elimina tablas de baselines en el schema de desarrollo
-- (creadas en 06a_setup_baselines.py)
-- NUEVO: Tablas genéricas compartidas entre modelos (sin nombre de modelo)

USE SCHEMA SC_STORAGE_BMX_PS;

-- Tablas de drift baselines
DROP TABLE IF EXISTS OBS_DATA_HIST_BL;
DROP TABLE IF EXISTS OBS_PRED_HIST_BL;
DROP TABLE IF EXISTS OBS_PERFORMANCE_BL;

-- Tablas de predicciones baseline
DROP TABLE IF EXISTS OBS_PREDICTIONS_BL;
DROP VIEW IF EXISTS OBS_PREDICTIONS_BL_VW;

-- Tablas auxiliares de baseline
DROP TABLE IF EXISTS TRAIN_CUST_CATEGORY_LOOKUP;
DROP VIEW IF EXISTS FEAT_CUSTBPR_WEEKLY__HOLDOUT_VW;

-- ============================================================================
-- SECCIÓN 5: TABLAS DE HIPERPARÁMETROS Y EXPERIMENTOS
-- ============================================================================
-- Elimina tabla de resultados de HPO (creada en 03_hyperparameter_search.py)

DROP TABLE IF EXISTS BD_AA_DEV.SC_MODELS_BMX.HPO_UNIBOX_CUSTBPR_WEEKLY_FORECAST;

-- NOTA: Los experimentos de ML (ML Experiments) no se pueden eliminar con DROP
-- Para limpiar experimentos, usa:
--   1. Snowsight UI: Data > ML Functions > Experiments
--   2. O el comando: DROP EXPERIMENT <experiment_name>
-- Experimentos creados:
--   - EXP_UNIBOX_CUSTBPR_WEEKLY_FORECAST_RANDOM_YYYYMMDD (script 03)
--   - EXP_UNIBOX_CUSTBPR_WEEKLY_FORECAST_BAYESIAN_YYYYMMDD (script 03b)

-- ============================================================================
-- SECCIÓN 6: FEATURE STORE COMPARTIDO
-- ============================================================================
-- Elimina tablas de features compartidas por entidad y frecuencia
-- NUEVO: Feature store compartido (sin nombre de modelo)

USE SCHEMA SC_FEATURES_BMX;

-- Tabla de features compartida (creada en 02_feature_store_setup.py)
-- NUEVO: FEAT_CUSTBPR_WEEKLY (entity + frequency, sin modelo)
DROP TABLE IF EXISTS FEAT_CUSTBPR_WEEKLY;

-- Tablas derivadas del feature store (creadas en 01_data_validation_and_cleaning.py)
DROP TABLE IF EXISTS FEAT_CUSTBPR_WEEKLY__TRAIN;
DROP TABLE IF EXISTS FEAT_CUSTBPR_WEEKLY__HOLDOUT;
DROP TABLE IF EXISTS FEAT_CUSTBPR_WEEKLY__INF;

-- Vistas del feature store (creadas por los scripts que las usan)
DROP VIEW IF EXISTS FEAT_CUSTBPR_WEEKLY__TRAIN_VW;
DROP VIEW IF EXISTS FEAT_CUSTBPR_WEEKLY__HOLDOUT_VW;
DROP VIEW IF EXISTS FEAT_CUSTBPR_WEEKLY__INF_VW;

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
-- FIN DEL SCRIPT DE LIMPIEZA V3
-- ============================================================================

-- NOTA FINAL:
-- Si después de ejecutar este script aún ves objetos residuales, puedes:
-- 1. Ejecutar los comandos SHOW de la sección 8 para identificarlos
-- 2. Eliminarlos manualmente con DROP
-- 3. Si hay problemas de dependencias, usa CASCADE:
--    DROP TABLE <nombre> CASCADE;
--
-- Versiones anteriores:
--   - cleanup_all_resources.sql (v1): Convención antigua (nombres originales)
--   - cleanup_all_resources_v2.sql (v2): Feature store 1:1 con modelo
--   - cleanup_all_resources_v3.sql (v3): Feature store compartido + OBS genérica
