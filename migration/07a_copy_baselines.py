# %% [markdown]
# # 07a — Copy Baselines to Production
#
# Copies baseline metrics and histogram tables from the development schema
# into the production schema.
#
# This is the “baselines” half of the old `07_environment_change.py`.

# %% [markdown]
# ## 1. Setup
#
# Initial setup: Snowpark session, source/target schema constants,
# baseline table names, and aggregation columns.

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F

session = get_active_session()

# %% [markdown]
# ### 1A. Constants
#
# Constants defining source (development) and target (production) schemas,
# the baseline tables to synchronize.

# %%
SRC_DATABASE = "BD_AA_DEV"
TGT_DATABASE = "BD_AA_DEV"

# Source schemas (development)
SRC_STORAGE_SCHEMA = "SC_STORAGE_BMX_PS"
SRC_MODELS_SCHEMA = "SC_MODELS_BMX"

# Target schemas (production)
TGT_STORAGE_SCHEMA = "SC_FEATURES_BMX"
TGT_MODELS_SCHEMA = "SC_STORAGE_BMX_PS"

# Model name (base for all model-specific objects)
MODEL_NAME = "UNIBOX_CUSTBPR_WEEKLY_FORECAST"

# Model FQNs (kept for parity with the original script; not used here)
SRC_MODEL_FQN = f"{SRC_DATABASE}.{SRC_MODELS_SCHEMA}.{MODEL_NAME}"
TGT_MODEL_FQN = f"{TGT_DATABASE}.{TGT_MODELS_SCHEMA}.{MODEL_NAME}"

# Baseline tables (source - generic, shared across models)
SRC_DATA_DRIFT_BASELINE = f"{SRC_DATABASE}.{SRC_STORAGE_SCHEMA}.OBS_DATA_HIST_BL"
SRC_PRED_DRIFT_BASELINE = f"{SRC_DATABASE}.{SRC_STORAGE_SCHEMA}.OBS_PRED_HIST_BL"
SRC_PERF_BASELINE = f"{SRC_DATABASE}.{SRC_STORAGE_SCHEMA}.OBS_PERFORMANCE_BL"

# Baseline tables (target - generic, shared across models)
TGT_DATA_DRIFT_BASELINE = f"{TGT_DATABASE}.{TGT_STORAGE_SCHEMA}.OBS_DATA_HIST_BL"
TGT_PRED_DRIFT_BASELINE = f"{TGT_DATABASE}.{TGT_STORAGE_SCHEMA}.OBS_PRED_HIST_BL"
TGT_PERF_BASELINE = f"{TGT_DATABASE}.{TGT_STORAGE_SCHEMA}.OBS_PERFORMANCE_BL"

session.sql(f"USE DATABASE {SRC_DATABASE}").collect()

# %% [markdown]
# ## 2. Create baseline tables if missing
#
# Create baseline tables in the production schema (if they don't exist)
# using the source table structure with CREATE TABLE ... LIKE.

# %%
baseline_pairs = [
    (SRC_DATA_DRIFT_BASELINE, TGT_DATA_DRIFT_BASELINE),
    (SRC_PRED_DRIFT_BASELINE, TGT_PRED_DRIFT_BASELINE),
    (SRC_PERF_BASELINE, TGT_PERF_BASELINE),
]

for src_tbl, tgt_tbl in baseline_pairs:
    session.sql(f"CREATE TABLE IF NOT EXISTS {tgt_tbl} LIKE {src_tbl}").collect()
    print(f"Table ready: {tgt_tbl}")

# %% [markdown]
# ## 3. Insert missing baseline data
#
# For each baseline table, identify combinations (MODEL_NAME, MODEL_VERSION,
# AGGREGATED_COL) that exist in the source but are missing in the target, and
# insert the corresponding rows.

# %%
sync_pairs = [
    (SRC_DATA_DRIFT_BASELINE, TGT_DATA_DRIFT_BASELINE),
    (SRC_PRED_DRIFT_BASELINE, TGT_PRED_DRIFT_BASELINE),
    (SRC_PERF_BASELINE, TGT_PERF_BASELINE),
]

for src_tbl, tgt_tbl in sync_pairs:
    src_combos = (
        session.table(src_tbl)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .select("MODEL_NAME", "MODEL_VERSION", "AGGREGATED_COL")
        .distinct()
    )

    tgt_combos = (
        session.table(tgt_tbl)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .select("MODEL_NAME", "MODEL_VERSION", "AGGREGATED_COL")
        .distinct()
    )

    missing_combos = src_combos.join(
        tgt_combos,
        on=["MODEL_NAME", "MODEL_VERSION", "AGGREGATED_COL"],
        how="left_anti",
    )

    missing_list = missing_combos.collect()

    if not missing_list:
        print(f"{tgt_tbl}: no missing combos, skipping.")
        continue

    for row in missing_list:
        mv = row["MODEL_VERSION"]
        ac = row["AGGREGATED_COL"]

        rows_to_insert = (
            session.table(src_tbl)
            .filter(F.col("MODEL_NAME") == MODEL_NAME)
            .filter(F.col("MODEL_VERSION") == mv)
            .filter(F.col("AGGREGATED_COL") == ac)
        )

        rows_to_insert.write.mode("append").save_as_table(tgt_tbl)
        count = rows_to_insert.count()
        print(
            f"  Inserted {count:,} rows into {tgt_tbl} for version={mv}, agg_col={ac}"
        )

    total = session.table(tgt_tbl).count()
    print(f"{tgt_tbl} now has {total:,} rows.")

# %% [markdown]
# ## Done
#
# Baseline data is now present in the production schemas.

