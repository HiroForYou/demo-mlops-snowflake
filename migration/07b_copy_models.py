# %% [markdown]
# # 07b — Copy Models and Apply Production Tags
#
# Copies the model version selected by the `PRODUCTION` alias in the
# development registry into the production registry and applies:
# - `PRODUCTION_<USE_CASE>` tag (active inference version)
# - `ROLLBACK_VERSION_<USE_CASE>` tag (previous active version, when available)
#
# This is the “model+tags” half of the old `07_environment_change.py`.

# %% [markdown]
# ## 1. Setup
#
# Initial setup: Snowpark session, source/target schema constants,
# and tag configuration.

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.ml.registry import Registry

session = get_active_session()

# %% [markdown]
# ### 1A. Constants
#
# Constants defining source (development) and target (production) schemas,
# model name, and tag naming.

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

# Model FQNs
SRC_MODEL_FQN = f"{SRC_DATABASE}.{SRC_MODELS_SCHEMA}.{MODEL_NAME}"
TGT_MODEL_FQN = f"{TGT_DATABASE}.{TGT_MODELS_SCHEMA}.{MODEL_NAME}"

# Use-case token used in production tags
USE_CASE = "CLIENTA_DEFAULT"  # Cambia segun cliente o caso de uso

# Tag for production model versions per use case
PRODUCTION_TAG_FQN = f"{TGT_DATABASE}.{TGT_MODELS_SCHEMA}.PRODUCTION_{USE_CASE}"
ROLLBACK_TAG_FQN = f"{TGT_DATABASE}.{TGT_MODELS_SCHEMA}.ROLLBACK_VERSION_{USE_CASE}"

session.sql(f"USE DATABASE {SRC_DATABASE}").collect()

# %% [markdown]
# ## 2. Migrate PRODUCTION model version
#
# Resolves the PRODUCTION alias to a specific version in the development
# schema and copies it to the production schema using CREATE MODEL /
# ALTER MODEL ADD VERSION.

# %%
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
    tgt_model_ref = None
    existing_versions = set()
    model_exists = False

if prod_version_name in existing_versions:
    print(
        f"Version {prod_version_name} already exists in {TGT_MODEL_FQN}, skipping migration."
    )
else:
    if model_exists:
        add_version_sql = f"""
        ALTER MODEL {TGT_MODEL_FQN} ADD VERSION {prod_version_name}
            FROM MODEL {SRC_MODEL_FQN} VERSION {prod_version_name}
        """
        session.sql(add_version_sql).collect()
        print(f"Added version {prod_version_name} to existing model {TGT_MODEL_FQN}")
    else:
        create_model_sql = f"""
        CREATE MODEL {TGT_MODEL_FQN} WITH VERSION {prod_version_name}
            FROM MODEL {SRC_MODEL_FQN} VERSION {prod_version_name}
        """
        session.sql(create_model_sql).collect()
        print(f"Created model {TGT_MODEL_FQN} with version {prod_version_name}")

# %% [markdown]
# ## 3. Refresh production/rollback tags
#
# Snapshot previous `PRODUCTION_<USE_CASE>` value into `ROLLBACK_VERSION_<USE_CASE>`
# (when model already existed and the previous tag is present), then set the new
# `PRODUCTION_<USE_CASE>` value to the promoted version.

# %%
session.sql(f"CREATE TAG IF NOT EXISTS {PRODUCTION_TAG_FQN}").collect()
session.sql(f"CREATE TAG IF NOT EXISTS {ROLLBACK_TAG_FQN}").collect()

previous_prod_version_name = None
if model_exists and tgt_model_ref is not None:
    try:
        desired_short = f"PRODUCTION_{USE_CASE}".upper()
        for tag_name, tag_value in tgt_model_ref.show_tags().items():
            if tag_name.split(".")[-1].upper() == desired_short:
                previous_prod_version_name = tag_value
                break
    except Exception:
        previous_prod_version_name = None

if previous_prod_version_name and previous_prod_version_name != prod_version_name:
    session.sql(f"""
        ALTER MODEL {TGT_MODEL_FQN}
            SET TAG {ROLLBACK_TAG_FQN} = '{previous_prod_version_name}'
    """).collect()
    print(
        f"Stored rollback: {ROLLBACK_TAG_FQN} = '{previous_prod_version_name}' on {TGT_MODEL_FQN}"
    )

session.sql(f"""
    ALTER MODEL {TGT_MODEL_FQN}
        SET TAG {PRODUCTION_TAG_FQN} = '{prod_version_name}'
""").collect()

print(f"Applied tag {PRODUCTION_TAG_FQN} = '{prod_version_name}' on {TGT_MODEL_FQN}")

# %% [markdown]
# ## Done
#
# The model version is now available in production and tagged for inference.

