# %% [markdown]
# # 06d — Performance baseline
#
# Part 4 (final) of the baseline generation pipeline.
# Calculates the baseline performance metrics (WAPE, RMSE, MAE, F1) by comparing
# baseline predictions with the actual training data.
# These are calculated both per segment and for the complete model ("full_model").
# These metrics serve as a reference to detect performance degradation in production.
#
# Prerequisite: 06a must have been executed so that PRED_BASELINE_VW exists,
# and 06b must have been executed so that DATA_DRIFT_HISTOGRAMS_BASELINE is populated.

# %% [markdown]
# ## 1. Setup

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F, Window
session = get_active_session()

# %% [markdown]
# ### 1A. Constants

# %%
# Account information
DATABASE = "BD_AA_DEV"
STORAGE_SCHEMA = "SC_STORAGE_BMX_PS"
FEATURES_SCHEMA = "SC_STORAGE_BMX_PS"

session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()

# Input data sources
FEATURE_TABLE = f"{DATABASE}.{FEATURES_SCHEMA}.TRAIN_DATASET_HOLDOUT_VW"

# Target tables (baseline only)
PERF_BASELINE = "DA_PERFORMANCE_BASELINE"
PRED_BASELINE_VW = "DA_PREDICTIONS_BASELINE_VW"

# Model specifications
MODEL_NAME = "UNI_BOX_REGRESSION_PARTITIONED"
ID_COLS = ["customer_id", "brand_pres_ret", "prod_key"]
AGG_COLS = ["STATS_NTILE_GROUP", "CUST_CATEGORY"]
TARGET_COL = "UNI_BOX_WEEK"
PREDICTION_COL = "PREDICTED_UNI_BOX_WEEK"
TIME_COL = "week"

# Performance metrics configuration
PERF_JOIN_KEYS = [*ID_COLS, TIME_COL]
PERF_METRIC_NAMES = ["wape", "rmse", "mae", "f1_binary"]

# %% [markdown]
# ### 1B. Functions

# %%
def get_new_baseline_versions(baseline_table, agg_col, metric_col):
    """Find model versions in PRED_BASELINE_VW that are missing from a baseline table.

    Parameters
    ----------
    baseline_table : str
        Fully qualified name of the baseline target table.
    agg_col : str
        Selected aggregation column.
    metric_col : str
        Selected metric.

    Returns
    -------
    list[str]
        Model versions that need baseline computation.
    """
    all_versions = (
        session.table(PRED_BASELINE_VW)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .select("MODEL_VERSION")
        .distinct()
    )

    existing_versions = (
        session.table(baseline_table)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(F.col("AGGREGATED_COL") == agg_col.lower())
        .filter(F.col("METRIC_COL") == metric_col)
        .select("MODEL_VERSION")
        .distinct()
    )

    new_versions = all_versions.join(
        existing_versions, on="MODEL_VERSION", how="left_anti"
    )

    result = [row["MODEL_VERSION"] for row in new_versions.collect()]
    print(f"{baseline_table} [{agg_col}] [{metric_col}]")
    print(f"  Versions needing baseline: {result}")
    return result

# %% [markdown]
# ## 5. Performance baseline
#
# Calculates the baseline performance metrics (WAPE, RMSE, MAE, F1) by comparing
# baseline predictions with the actual training data.
# These are calculated both per segment and for the complete model ("full_model").
# These metrics serve as a reference to detect performance degradation in production.

# %%
def compute_performance_metrics(paired_df, agg_col):
    """Compute WAPE, RMSE, MAE and F1 from a paired predictions+actuals DataFrame.

    Parameters
    ----------
    paired_df : snowpark.DataFrame
        Must contain columns: WEEK, MODEL_VERSION, TARGET_COL, PREDICTION_COL,
        and the aggregation column *agg_col*.
    agg_col : str
        Column to group by (e.g. "STATS_NTILE_GROUP").

    Returns
    -------
    snowpark.DataFrame
        Long format with columns: WEEK, MODEL_VERSION, AGGREGATED_COL,
        AGGREGATED_VALUE, METRIC_COL, METRIC_VALUE.
    """
    agg_col_lower = agg_col.lower()
    actual = F.col(TARGET_COL)
    predicted = F.col(PREDICTION_COL)
    abs_error = F.abs(actual - predicted)
    sq_error = F.pow(actual - predicted, F.lit(2))
    actual_positive = F.iff(actual > F.lit(0), F.lit(1), F.lit(0))
    predicted_positive = F.iff(predicted > F.lit(0), F.lit(1), F.lit(0))

    def _metrics_for_group(df, group_cols, agg_value_expr):
        """Compute the 4 metrics for a given grouping."""
        base = df.group_by(*group_cols).agg(
            # WAPE components
            F.sum(abs_error).alias("SUM_ABS_ERROR"),
            F.sum(F.abs(actual)).alias("SUM_ABS_ACTUAL"),
            # RMSE
            F.avg(sq_error).alias("AVG_SQ_ERROR"),
            # MAE
            F.avg(abs_error).alias("MAE_VAL"),
            # F1 components
            F.sum(actual_positive * predicted_positive).alias("TP"),
            F.sum((F.lit(1) - actual_positive) * predicted_positive).alias("FP"),
            F.sum(actual_positive * (F.lit(1) - predicted_positive)).alias("FN"),
        )

        wape = base.select(
            F.col(TIME_COL), F.col("MODEL_VERSION"),
            F.lit(agg_col_lower).alias("AGGREGATED_COL"),
            agg_value_expr.alias("AGGREGATED_VALUE"),
            F.lit("wape").alias("METRIC_COL"),
            (F.col("SUM_ABS_ERROR") / F.greatest(F.col("SUM_ABS_ACTUAL"), F.lit(1e-10)))
                .cast("DOUBLE").alias("METRIC_VALUE"),
        )

        rmse = base.select(
            F.col(TIME_COL), F.col("MODEL_VERSION"),
            F.lit(agg_col_lower).alias("AGGREGATED_COL"),
            agg_value_expr.alias("AGGREGATED_VALUE"),
            F.lit("rmse").alias("METRIC_COL"),
            F.sqrt(F.col("AVG_SQ_ERROR")).cast("DOUBLE").alias("METRIC_VALUE"),
        )

        mae = base.select(
            F.col(TIME_COL), F.col("MODEL_VERSION"),
            F.lit(agg_col_lower).alias("AGGREGATED_COL"),
            agg_value_expr.alias("AGGREGATED_VALUE"),
            F.lit("mae").alias("METRIC_COL"),
            F.col("MAE_VAL").cast("DOUBLE").alias("METRIC_VALUE"),
        )

        precision = F.col("TP") / F.greatest(F.col("TP") + F.col("FP"), F.lit(1e-10))
        recall = F.col("TP") / F.greatest(F.col("TP") + F.col("FN"), F.lit(1e-10))

        f1 = base.select(
            F.col(TIME_COL), F.col("MODEL_VERSION"),
            F.lit(agg_col_lower).alias("AGGREGATED_COL"),
            agg_value_expr.alias("AGGREGATED_VALUE"),
            F.lit("f1_binary").alias("METRIC_COL"),
            (F.lit(2) * precision * recall / F.greatest(precision + recall, F.lit(1e-10)))
                .cast("DOUBLE").alias("METRIC_VALUE"),
        )

        return wape.union_all_by_name(rmse).union_all_by_name(mae).union_all_by_name(f1)

    # Per aggregation value
    per_segment = _metrics_for_group(
        paired_df,
        [TIME_COL, "MODEL_VERSION", agg_col],
        F.col(agg_col).cast("STRING"),
    )

    # Full model (across all aggregation values)
    full_model = _metrics_for_group(
        paired_df,
        [TIME_COL, "MODEL_VERSION"],
        F.lit("full_model"),
    )

    return per_segment.union_all_by_name(full_model)

# %%
# Compute baseline performance metrics -- iterate over each aggregation column

for agg_col in AGG_COLS:
    new_versions = get_new_baseline_versions(
        PERF_BASELINE, agg_col=agg_col, metric_col=PERF_METRIC_NAMES[0])

    if not new_versions:
        print(f"  No new baseline versions for {agg_col}, skipping.")
        continue

    for version in new_versions:
        # Join baseline predictions with baseline actuals.
        # PRED_BASELINE_VW stores join keys inside ENTITY_MAP, so we extract them.
        preds = (
            session.table(PRED_BASELINE_VW)
            .filter(F.col("MODEL_NAME") == MODEL_NAME)
            .filter(F.col("MODEL_VERSION") == version)
            .with_column("CUSTOMER_ID", F.col("ENTITY_MAP")["customer_id"].cast("STRING"))
            .with_column(TIME_COL, F.col("ENTITY_MAP")[TIME_COL].cast("STRING"))
            .with_column("BRAND_PRES_RET", F.col("ENTITY_MAP")["brand_pres_ret"].cast("STRING"))
            .with_column("PROD_KEY", F.col("ENTITY_MAP")["prod_key"].cast("STRING"))
            .with_column(PREDICTION_COL, F.col("PREDICTION"))
        )
        actuals = session.table(FEATURE_TABLE).drop(AGG_COLS)

        paired = preds.join(actuals, on=PERF_JOIN_KEYS, how="inner")

        # Compute metrics
        metrics_df = compute_performance_metrics(paired, agg_col)

        # Format according to the PERF_BASELINE schema
        data_date_expr = F.dateadd(
            "week", F.col(TIME_COL) % F.lit(100) - F.lit(1),
            F.date_from_parts(F.floor(F.col(TIME_COL) / F.lit(100)), F.lit(1), F.lit(1)),
        )

        baseline_rows = (
            metrics_df
            .with_column("DATA_DATE", data_date_expr)
            .select(
                F.sha2(
                    F.concat(
                        F.lit(MODEL_NAME), F.lit("||"),
                        F.col("MODEL_VERSION"), F.lit("||"),
                        F.col(TIME_COL), F.lit("||"),
                        F.col("AGGREGATED_COL"), F.lit("||"),
                        F.col("AGGREGATED_VALUE"), F.lit("||"),
                        F.col("METRIC_COL"),
                    ),
                    256,
                ).alias("RECORD_ID"),
                F.lit(MODEL_NAME).alias("MODEL_NAME"),
                F.col("MODEL_VERSION"),
                F.object_construct(
                    F.lit(TIME_COL), F.col(TIME_COL),
                    F.lit("data_date"), F.col("DATA_DATE"),
                ).alias("ENTITY_MAP"),
                F.col("AGGREGATED_COL"),
                F.col("AGGREGATED_VALUE"),
                F.col("METRIC_COL"),
                F.col("METRIC_VALUE"),
                F.lit("MXBEB").alias("BKCC"),
                F.to_char(F.col("DATA_DATE"), F.lit("YYYYMM")).alias("CALMONTH"),
                F.current_timestamp().alias("LDTS"),
            )
        )

        baseline_rows.write.mode("append").save_as_table(PERF_BASELINE)

    n_rows = session.table(PERF_BASELINE).count()
    print(f"Baseline performance for {agg_col}: PERF_BASELINE now has {n_rows} records.")
