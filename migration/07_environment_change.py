# %% [markdown]
# # Environment change
#
# This notebook migrates the model and its baseline data from the development schema
# to the production schema. It is the intermediate step between creating baselines (06)
# and production inference (08). It copies the PRODUCTION version of the model, replicas the
# baseline metrics and histogram tables, and applies a CANDIDATE_VERSION tag to the model
# in the target schema so the inference notebook knows which version to execute.

# %% [markdown]
# ## 1. Setup
#
# Initial setup: Snowpark session, source/target schema constants,
# baseline table names, and aggregation columns.

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F
from snowflake.ml.registry import Registry
session = get_active_session()

# %% [markdown]
# ### 1A. Constants
#
# Constants defining source (development) and target (production) schemas,
# the baseline tables to synchronize, and the CANDIDATE_VERSION tag that marks
# the model version ready for inference.

# %%
SRC_DATABASE = "BD_AA_DEV"
TGT_DATABASE = "BD_AA_DEV"

# Source schemas (development)
SRC_STORAGE_SCHEMA = "SC_STORAGE_BMX_PS"
SRC_MODELS_SCHEMA = "SC_MODELS_BMX"

# Target schemas (production)
TGT_STORAGE_SCHEMA = "SC_FEATURES_BMX"
TGT_MODELS_SCHEMA = "SC_STORAGE_BMX_PS"

# Model
MODEL_NAME = "UNI_BOX_REGRESSION_PARTITIONED"
SRC_MODEL_FQN = f"{SRC_DATABASE}.{SRC_MODELS_SCHEMA}.{MODEL_NAME}"
TGT_MODEL_FQN = f"{TGT_DATABASE}.{TGT_MODELS_SCHEMA}.{MODEL_NAME}"

# Baseline tables (source)
SRC_DATA_DRIFT_BASELINE = f"{SRC_DATABASE}.{SRC_STORAGE_SCHEMA}.DA_DATA_DRIFT_HISTOGRAMS_BASELINE"
SRC_PRED_DRIFT_BASELINE = f"{SRC_DATABASE}.{SRC_STORAGE_SCHEMA}.DA_PREDICTION_DRIFT_HISTOGRAMS_BASELINE"
SRC_PERF_BASELINE = f"{SRC_DATABASE}.{SRC_STORAGE_SCHEMA}.DA_PERFORMANCE_BASELINE"

# Baseline tables (target)
TGT_DATA_DRIFT_BASELINE = f"{TGT_DATABASE}.{TGT_STORAGE_SCHEMA}.DA_DATA_DRIFT_HISTOGRAMS_BASELINE"
TGT_PRED_DRIFT_BASELINE = f"{TGT_DATABASE}.{TGT_STORAGE_SCHEMA}.DA_PREDICTION_DRIFT_HISTOGRAMS_BASELINE"
TGT_PERF_BASELINE = f"{TGT_DATABASE}.{TGT_STORAGE_SCHEMA}.DA_PERFORMANCE_BASELINE"

# Tag for candidate model versions
CANDIDATE_TAG_FQN = f"{TGT_DATABASE}.{TGT_MODELS_SCHEMA}.CANDIDATE_VERSION"

# Aggregation columns used in histogram/perf tables
AGG_COLS = ["STATS_NTILE_GROUP", "CUST_CATEGORY"]

# Performance metric names (used to check missing combos in perf table)
PERF_METRIC_NAMES = ["wape", "rmse", "mae", "f1_binary"]

session.sql(f"USE DATABASE {SRC_DATABASE}").collect()

# %% [markdown]
# ## 2. Migrate PRODUCTION model version
#
# Resolves the PRODUCTION alias to a specific version in the development schema and
# copies it to the production schema using CREATE MODEL / ALTER MODEL ADD VERSION.
# If the version already exists in the target, migration is skipped.
# Then applies the CANDIDATE_VERSION tag to the model to signal which version should
# be used for inference (notebook 08).

# %%
# Use the PRODUCTION alias to identify the relevant model version
src_registry = Registry(
    session=session,
    database_name=SRC_DATABASE,
    schema_name=SRC_MODELS_SCHEMA,
)
src_model_ref = src_registry.get_model(MODEL_NAME)
prod_version = src_model_ref.version("PRODUCTION")
prod_version_name = prod_version.version_name

print(f"Source PRODUCTION alias -> version: {prod_version_name}")

# %%
# Check if the model already exists in the other environment. If not, copy the PRODUCTION 
# version; otherwise, add the PRODUCTION version to the model.

tgt_registry = Registry(
    session=session,
    database_name=TGT_DATABASE,
    schema_name=TGT_MODELS_SCHEMA,
)

try:
    tgt_model_ref = tgt_registry.get_model(MODEL_NAME)
    existing_versions = {v.version_name for v in tgt_model_ref.versions()}
    model_exists = True
except Exception:
    existing_versions = set()
    model_exists = False

if prod_version_name in existing_versions:
    print(f"Version {prod_version_name} already exists in {TGT_MODEL_FQN}, skipping migration.")
else:
    if model_exists:
        # Model object exists but this version is missing — add the version
        add_version_sql = f"""
        ALTER MODEL {TGT_MODEL_FQN} ADD VERSION {prod_version_name}
            FROM MODEL {SRC_MODEL_FQN} VERSION {prod_version_name}
        """
        session.sql(add_version_sql).collect()
        print(f"Added version {prod_version_name} to existing model {TGT_MODEL_FQN}")
    else:
        # Model does not exist — create it with this version
        create_model_sql = f"""
        CREATE MODEL {TGT_MODEL_FQN} WITH VERSION {prod_version_name}
            FROM MODEL {SRC_MODEL_FQN} VERSION {prod_version_name}
        """
        session.sql(create_model_sql).collect()
        print(f"Created model {TGT_MODEL_FQN} with version {prod_version_name}")

# %%
# Create the tag if it doesn't exist and apply it to the model
session.sql(f"CREATE TAG IF NOT EXISTS {CANDIDATE_TAG_FQN}").collect()

session.sql(f"""
    ALTER MODEL {TGT_MODEL_FQN}
        SET TAG {CANDIDATE_TAG_FQN} = '{prod_version_name}'
""").collect()

print(f"Applied tag {CANDIDATE_TAG_FQN} = '{prod_version_name}' on {TGT_MODEL_FQN}")

# %% [markdown]
# ## 3. Create baseline tables if missing
#
# Create baseline tables in the production schema (if they don't exist)
# using the source table structure with CREATE TABLE ... LIKE.

# %%
# For each baseline table, create the target table LIKE the source if it doesn't exist.

baseline_pairs = [
    (SRC_DATA_DRIFT_BASELINE, TGT_DATA_DRIFT_BASELINE),
    (SRC_PRED_DRIFT_BASELINE, TGT_PRED_DRIFT_BASELINE),
    (SRC_PERF_BASELINE, TGT_PERF_BASELINE),
]

for src_tbl, tgt_tbl in baseline_pairs:
    session.sql(f"CREATE TABLE IF NOT EXISTS {tgt_tbl} LIKE {src_tbl}").collect()
    print(f"Table ready: {tgt_tbl}")

# %% [markdown]
# ## 4. Insert missing baseline data
#
# For each baseline table, identify combinations (MODEL_NAME, MODEL_VERSION,
# AGGREGATED_COL) that exist in the source but are missing in the target, and insert the
# corresponding rows. This ensures that the production schema has all the
# reference data needed for monitoring (notebook 09).

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
        print(f"  Inserted {count:,} rows into {tgt_tbl} for version={mv}, agg_col={ac}")

    total = session.table(tgt_tbl).count()
    print(f"{tgt_tbl} now has {total:,} rows.")

# %% [markdown]
# ## Done
#
# The model version and baseline data have been migrated to the production schemas.
# The next step is to run inference on production data (notebook 08).

