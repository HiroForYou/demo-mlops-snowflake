# %% [markdown]
# # Partitioned Inference — UNI_BOX_REGRESSION_PARTITIONED
#
# Inference usando alias **PRODUCTION** directamente, con SAMPLE opcional.

# %%
from snowflake.snowpark.context import get_active_session
import time

# %%
INFERENCE_SAMPLE_FRACTION = 0.01

DATABASE = "BD_AA_DEV"
STORAGE_SCHEMA = "SC_STORAGE_BMX_PS"
MODEL_SCHEMA = "SC_MODELS_BMX"

SOURCE_TABLE = f"{DATABASE}.{STORAGE_SCHEMA}.INFERENCE_INPUT_TEMP"
MODEL_FQN = f"{DATABASE}.{MODEL_SCHEMA}.UNI_BOX_REGRESSION_PARTITIONED"

# %%
session = get_active_session()
session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()

print("✅ Connected to Snowflake")
print("   Model: UNI_BOX_REGRESSION_PARTITIONED (PRODUCTION alias)")

# %% [markdown]
# ## Input sampling

# %%
if INFERENCE_SAMPLE_FRACTION is not None and 0 < INFERENCE_SAMPLE_FRACTION < 1:
    FROM_SQL = (
        f"{SOURCE_TABLE} i "
        f"SAMPLE BERNOULLI ({INFERENCE_SAMPLE_FRACTION * 100})"
    )
    print(f"⚠️  Using SAMPLE: {INFERENCE_SAMPLE_FRACTION*100:.2f}%")
else:
    FROM_SQL = f"{SOURCE_TABLE} i"
    print("✅ Using FULL dataset")

# %% [markdown]
# ## Partitioned inference (single-pass)

# %%
print("🚀 RUNNING PARTITIONED INFERENCE (single-pass)")
start_time = time.time()

inference_sql = f"""
WITH typed_input AS (
    SELECT
        i.CUSTOMER_ID::VARCHAR           AS CUSTOMER_ID,
        i.STATS_NTILE_GROUP::VARCHAR     AS STATS_NTILE_GROUP,
        i.WEEK::VARCHAR                  AS WEEK,
        i.BRAND_PRES_RET::VARCHAR        AS BRAND_PRES_RET,

        i.SUM_PAST_12_WEEKS::FLOAT       AS SUM_PAST_12_WEEKS,
        i.AVG_PAST_12_WEEKS::FLOAT       AS AVG_PAST_12_WEEKS,
        i.MAX_PAST_24_WEEKS::FLOAT       AS MAX_PAST_24_WEEKS,
        i.SUM_PAST_24_WEEKS::FLOAT       AS SUM_PAST_24_WEEKS,
        i.WEEK_OF_YEAR::NUMBER           AS WEEK_OF_YEAR,
        i.AVG_AVG_DAILY_ALL_HOURS::FLOAT AS AVG_AVG_DAILY_ALL_HOURS,
        i.SUM_P4W::FLOAT                 AS SUM_P4W,
        i.AVG_PAST_24_WEEKS::FLOAT       AS AVG_PAST_24_WEEKS,
        i.PHARM_SUPER_CONV::NUMBER       AS PHARM_SUPER_CONV,
        i.WINES_LIQUOR::NUMBER           AS WINES_LIQUOR,
        i.GROCERIES::NUMBER              AS GROCERIES,
        i.MAX_PREV2::FLOAT               AS MAX_PREV2,
        i.AVG_PREV2::FLOAT               AS AVG_PREV2,
        i.MAX_PREV3::FLOAT               AS MAX_PREV3,
        i.AVG_PREV3::FLOAT               AS AVG_PREV3,
        i.W_M1_TOTAL::FLOAT              AS W_M1_TOTAL,
        i.W_M2_TOTAL::FLOAT              AS W_M2_TOTAL,
        i.W_M3_TOTAL::FLOAT              AS W_M3_TOTAL,
        i.W_M4_TOTAL::FLOAT              AS W_M4_TOTAL,
        i.SPEC_FOODS::NUMBER             AS SPEC_FOODS,
        i.NUM_COOLERS::FLOAT             AS NUM_COOLERS,
        i.NUM_DOORS::NUMBER              AS NUM_DOORS,
        i.MAX_PAST_4_WEEKS::FLOAT        AS MAX_PAST_4_WEEKS,
        i.SUM_PAST_4_WEEKS::FLOAT        AS SUM_PAST_4_WEEKS,
        i.AVG_PAST_4_WEEKS::FLOAT        AS AVG_PAST_4_WEEKS,
        i.MAX_PAST_12_WEEKS::FLOAT       AS MAX_PAST_12_WEEKS
    FROM {FROM_SQL}
)
SELECT
    p.CUSTOMER_ID,
    p.STATS_NTILE_GROUP,
    p.WEEK,
    p.BRAND_PRES_RET,
    p.predicted_uni_box_week
FROM typed_input t,
TABLE(
  MODEL({MODEL_FQN}, PRODUCTION)!PREDICT(
    t.CUSTOMER_ID,
    t.STATS_NTILE_GROUP,
    t.WEEK,
    t.BRAND_PRES_RET,
    t.SUM_PAST_12_WEEKS,
    t.AVG_PAST_12_WEEKS,
    t.MAX_PAST_24_WEEKS,
    t.SUM_PAST_24_WEEKS,
    t.WEEK_OF_YEAR,
    t.AVG_AVG_DAILY_ALL_HOURS,
    t.SUM_P4W,
    t.AVG_PAST_24_WEEKS,
    t.PHARM_SUPER_CONV,
    t.WINES_LIQUOR,
    t.GROCERIES,
    t.MAX_PREV2,
    t.AVG_PREV2,
    t.MAX_PREV3,
    t.AVG_PREV3,
    t.W_M1_TOTAL,
    t.W_M2_TOTAL,
    t.W_M3_TOTAL,
    t.W_M4_TOTAL,
    t.SPEC_FOODS,
    t.NUM_COOLERS,
    t.NUM_DOORS,
    t.MAX_PAST_4_WEEKS,
    t.SUM_PAST_4_WEEKS,
    t.AVG_PAST_4_WEEKS,
    t.MAX_PAST_12_WEEKS
  ) OVER (PARTITION BY t.STATS_NTILE_GROUP)
) p
"""

# %%
df = session.sql(inference_sql)
count = df.count()
elapsed = time.time() - start_time

print(f"✅ {count:,} predictions in {elapsed:.2f}s")
df.show(10)
