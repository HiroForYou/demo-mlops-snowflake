---------------------------------------------------------------
-- 1. Before of 06-create-baselines
USE SCHEMA BD_AA_DEV.SC_STORAGE_BMX_PS;

CREATE or replace TRANSIENT TABLE train_cust_category_lookup AS
SELECT DISTINCT
    customer_id,
    brand_pres_ret,
    week,
    stats_ntile_group,
    CASE 
        WHEN pharm_super_conv = 1 THEN 'pharm_super_conv'
        WHEN wines_liquor = 1 THEN 'wines_liquor'
        WHEN groceries = 1 THEN 'groceries'
        WHEN spec_foods = 1 THEN 'spec_foods'
        ELSE 'others'
        END AS cust_category
FROM train_dataset_holdout
;

CREATE OR REPLACE VIEW train_dataset_holdout_vw AS
SELECT vw.*, mp.cust_category
FROM train_dataset_holdout AS vw
LEFT JOIN train_cust_category_lookup AS mp
ON vw.customer_id = mp.customer_id
    AND vw.week = mp.week
    AND vw.brand_pres_ret = mp.brand_pres_ret
    AND vw.stats_ntile_group = mp.stats_ntile_group
;

---------------------------------------------------------------------
-- 2. After of 06-create-baselines
-- Should be a view but this join takes *forever*
CREATE OR REPLACE TRANSIENT TABLE da_predictions_baseline_vw AS
SELECT
    vw.*,
    mp.stats_ntile_group,
    mp.cust_category
FROM DA_PREDICTIONS_BASELINE AS vw
LEFT JOIN train_cust_category_lookup AS mp
ON vw.entity_map:customer_id = mp.customer_id
    AND vw.entity_map:brand_pres_ret = mp.brand_pres_ret
    AND vw.entity_map:week = mp.week
    AND vw.entity_map:partition_value = mp.stats_ntile_group
;

---------------------------------------------------------------------
-- 3. Before of 08-partitioned-inference-batch
-- SCHEMA SC_FEATURES_BMX
CREATE OR REPLACE TRANSIENT TABLE BD_AA_DEV.SC_FEATURES_BMX.inference_dataset_cleaned
CLONE BD_AA_DEV.SC_STORAGE_BMX_PS.inference_dataset_cleaned;

CREATE or replace TRANSIENT TABLE BD_AA_DEV.SC_FEATURES_BMX.inference_cust_category_lookup AS
SELECT DISTINCT
    customer_id,
    brand_pres_ret,
    week,
    stats_ntile_group,
    CASE 
        WHEN pharm_super_conv = 1 THEN 'pharm_super_conv'
        WHEN wines_liquor = 1 THEN 'wines_liquor'
        WHEN groceries = 1 THEN 'groceries'
        WHEN spec_foods = 1 THEN 'spec_foods'
        ELSE 'others'
        END AS cust_category
FROM BD_AA_DEV.SC_FEATURES_BMX.inference_dataset_cleaned
; 

CREATE OR REPLACE VIEW BD_AA_DEV.SC_FEATURES_BMX.inference_dataset_cleaned_vw AS
SELECT vw.*, mp.cust_category
FROM BD_AA_DEV.SC_FEATURES_BMX.inference_dataset_cleaned AS vw
LEFT JOIN BD_AA_DEV.SC_FEATURES_BMX.inference_cust_category_lookup AS mp
ON vw.customer_id = mp.customer_id
    AND vw.week = mp.week
    AND vw.brand_pres_ret = mp.brand_pres_ret
;

---------------------------------------------------------------------
-- 4. After of 08-partitioned-inference-batch
-- Should be a view but this join takes *forever*
CREATE OR REPLACE TRANSIENT TABLE BD_AA_DEV.SC_FEATURES_BMX.da_predictions_vw AS
SELECT
    vw.*,
    mp.stats_ntile_group,
    mp.cust_category
FROM BD_AA_DEV.SC_FEATURES_BMX.da_predictions AS vw
LEFT JOIN BD_AA_DEV.SC_FEATURES_BMX.inference_cust_category_lookup AS mp
ON vw.entity_map:customer_id = mp.customer_id
    AND vw.entity_map:brand_pres_ret = mp.brand_pres_ret
    AND vw.entity_map:week = mp.week
;

CREATE OR REPLACE TRANSIENT TABLE BD_AA_DEV.SC_FEATURES_BMX.ground_truth_dataset_structured
CLONE ground_truth_dataset_structured
;

CREATE OR REPLACE VIEW BD_AA_DEV.SC_FEATURES_BMX.actuals_table_vw AS
SELECT *
FROM BD_AA_DEV.SC_FEATURES_BMX.ground_truth_dataset_structured AS td
WHERE week < 202548
;



-- -- Created in "DEV"
-- DROP TABLE BD_AA_DEV.SC_STORAGE_BMX_PS.DA_PREDICTIONS_BASELINE;
-- DROP TABLE BD_AA_DEV.SC_STORAGE_BMX_PS.DA_DATA_DRIFT_HISTOGRAMS_BASELINE;
-- DROP TABLE BD_AA_DEV.SC_STORAGE_BMX_PS.DA_PERFORMANCE_BASELINE;
-- DROP TABLE BD_AA_DEV.SC_STORAGE_BMX_PS.DA_PREDICTION_DRIFT_HISTOGRAMS_BASELINE;

-- DROP TABLE BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_CUST_CATEGORY_LOOKUP;
-- DROP VIEW BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_HOLDOUT_VW;
-- DROP TABLE BD_AA_DEV.SC_STORAGE_BMX_PS.DA_PREDICTIONS_BASELINE_VW;

-- -- Copied from "DEV" to "PROD"
-- DROP TABLE BD_AA_DEV.SC_FEATURES_BMX.DA_DATA_DRIFT_HISTOGRAMS_BASELINE;
-- DROP TABLE BD_AA_DEV.SC_FEATURES_BMX.DA_PERFORMANCE_BASELINE;
-- DROP TABLE BD_AA_DEV.SC_FEATURES_BMX.DA_PREDICTION_DRIFT_HISTOGRAMS_BASELINE;

-- -- Created in "PROD"
-- DROP TABLE BD_AA_DEV.SC_FEATURES_BMX.DA_PREDICTIONS;
-- DROP TABLE BD_AA_DEV.SC_FEATURES_BMX.DA_DATA_DRIFT_HISTOGRAMS;
-- DROP TABLE BD_AA_DEV.SC_FEATURES_BMX.DA_DATA_DRIFT; 
-- DROP TABLE BD_AA_DEV.SC_FEATURES_BMX.DA_PREDICTION_DRIFT_HISTOGRAMS;
-- DROP TABLE BD_AA_DEV.SC_FEATURES_BMX.DA_PREDICTION_DRIFT;
-- DROP TABLE BD_AA_DEV.SC_FEATURES_BMX.DA_PERFORMANCE;

-- DROP TABLE BD_AA_DEV.SC_FEATURES_BMX.INFERENCE_CUST_CATEGORY_LOOKUP;
-- DROP TABLE BD_AA_DEV.SC_FEATURES_BMX.INFERENCE_DATASET_CLEANED;
-- DROP TABLE BD_AA_DEV.SC_FEATURES_BMX.GROUND_TRUTH_DATASET_STRUCTURED;
-- DROP TABLE BD_AA_DEV.SC_FEATURES_BMX.DA_PREDICTIONS_VW;
-- DROP VIEW BD_AA_DEV.SC_FEATURES_BMX.ACTUALS_TABLE_VW;
