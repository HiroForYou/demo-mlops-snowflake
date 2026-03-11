# %% [markdown]
# # 09a — Setup: Landing tables
#
# Part 1 of the ML Observability pipeline.
# Creates landing tables for drift histograms (data and predictions),
# drift metrics (with thresholds and alert levels), and performance metrics.
# Run this script once before orchestrating the remaining subscripts (09b, 09c, 09d).

# %% [markdown]
# ## 1. Setup
#
# Initial setup: Snowpark session, constants, landing table creation,
# and auxiliary functions to identify new combinations to process.

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F, Window, Row
session = get_active_session()

# %% [markdown]
# ### 1A. Constants
#
# Project constants: input tables (features, predictions, actuals),
# baseline tables (read-only, populated by notebooks 06+07), landing tables
# (created by this notebook), and drift metrics and thresholds configuration.

# %%
# Account info
DATABASE = "BD_AA_DEV"
STORAGE_SCHEMA = "SC_FEATURES_BMX"
FEATURES_SCHEMA = "SC_FEATURES_BMX"

session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()

# Input data sources (all in SC_FEATURES_BMX)
FEATURE_TABLE = f"{DATABASE}.{FEATURES_SCHEMA}.INFERENCE_DATASET_CLEANED_VW"
PREDICTION_TABLE = "DA_PREDICTIONS_VW"
ACTUALS_TABLE = "ACTUALS_TABLE_VW"

# Baseline tables (read-only, populated by 16+17)
DATA_DRIFT_HISTOGRAMS_BASELINE = f"DA_DATA_DRIFT_HISTOGRAMS_BASELINE"
PRED_DRIFT_HISTOGRAMS_BASELINE = f"DA_PREDICTION_DRIFT_HISTOGRAMS_BASELINE"
PERF_BASELINE = f"DA_PERFORMANCE_BASELINE"

# Landing tables (created by this notebook)
DATA_DRIFT_HISTOGRAMS = f"DA_DATA_DRIFT_HISTOGRAMS"
DATA_DRIFT = f"DA_DATA_DRIFT"
PRED_DRIFT_HISTOGRAMS = f"DA_PREDICTION_DRIFT_HISTOGRAMS"
PRED_DRIFT = f"DA_PREDICTION_DRIFT"
PERF_DRIFT = f"DA_PERFORMANCE"

# Model specifics
MODEL_NAME = "UNI_BOX_REGRESSION_PARTITIONED"
ID_COLS = ["customer_id", "brand_pres_ret", "prod_key"]
AGG_COLS = ["STATS_NTILE_GROUP", "CUST_CATEGORY"]
TARGET_COL = "ACTUAL_SALES"
PREDICTION_COL = "PREDICTED_UNI_BOX_WEEK"
TIME_COL = "week"

# Feature drift settings
N_BINS = 20

# Performance metric settings
PERF_JOIN_KEYS = ["CUSTOMER_ID", "BRAND_PRES_RET", TIME_COL]  # actuals table has no PROD_KEY
PERF_METRIC_NAMES = ["wape", "rmse", "mae", "f1_binary"]

# %% [markdown]
# ### 1B. Create landing tables
#
# Creates landing tables for drift histograms (data and predictions),
# drift metrics (with thresholds and alert levels), and performance metrics.

# %%
HISTOGRAM_SCHEMA = """
    RECORD_ID        VARCHAR(128) NOT NULL,
    MODEL_NAME       VARCHAR(64)  NOT NULL,
    MODEL_VERSION    VARCHAR(32)  NOT NULL,
    ENTITY_MAP       OBJECT,
    AGGREGATED_COL   VARCHAR(64)  NOT NULL,
    AGGREGATED_VALUE VARCHAR(64),
    METRIC_COL       VARCHAR(64)  NOT NULL,
    METRIC_MAP       OBJECT,
    CALMONTH         VARCHAR(6),
    LDTS             TIMESTAMP_LTZ(9) NOT NULL
"""

for tbl in [DATA_DRIFT_HISTOGRAMS, PRED_DRIFT_HISTOGRAMS]:
    session.sql(f"CREATE TABLE IF NOT EXISTS {tbl} ({HISTOGRAM_SCHEMA})").collect()


DRIFT_SCHEMA = """
    RECORD_ID          VARCHAR(128) NOT NULL,
    MODEL_NAME         VARCHAR(64)  NOT NULL,
    MODEL_VERSION      VARCHAR(32)  NOT NULL,
    ENTITY_MAP         OBJECT,
    AGGREGATED_COL     VARCHAR(64)  NOT NULL,
    AGGREGATED_VALUE   VARCHAR(64)  NOT NULL,
    METRIC_COL         VARCHAR(64)  NOT NULL,
    METRIC_VALUE       FLOAT,
    WARNING_THRESHOLD  FLOAT        NOT NULL,
    CRITICAL_THRESHOLD FLOAT        NOT NULL,
    ALERT_LEVEL        INT NOT NULL,
    BKCC               VARCHAR(5)   NOT NULL,
    CALMONTH           VARCHAR(6),
    LDTS               TIMESTAMP_LTZ(9) NOT NULL
"""

for tbl in [DATA_DRIFT, PRED_DRIFT]:
    session.sql(f"CREATE TABLE IF NOT EXISTS {tbl} ({DRIFT_SCHEMA})").collect()


PERF_SCHEMA = """
    RECORD_ID          VARCHAR(128) NOT NULL,
    MODEL_NAME         VARCHAR(64)  NOT NULL,
    MODEL_VERSION      VARCHAR(32)  NOT NULL,
    ENTITY_MAP         OBJECT,
    AGGREGATED_COL     VARCHAR(64)  NOT NULL,
    AGGREGATED_VALUE   VARCHAR(64)  NOT NULL,
    METRIC_COL         VARCHAR(64)  NOT NULL,
    METRIC_VALUE       FLOAT,
    METRIC_DRIFT       FLOAT,
    WARNING_THRESHOLD  FLOAT        NOT NULL,
    CRITICAL_THRESHOLD FLOAT        NOT NULL,
    ALERT_LEVEL        INT NOT NULL,
    BKCC               VARCHAR(5)   NOT NULL,
    CALMONTH           VARCHAR(6),
    LDTS               TIMESTAMP_LTZ(9) NOT NULL
"""

session.sql(f"CREATE TABLE IF NOT EXISTS {PERF_DRIFT} ({PERF_SCHEMA})").collect()

print("Landing tables ready.")
