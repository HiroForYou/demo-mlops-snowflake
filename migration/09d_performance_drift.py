# %% [markdown]
# # 09d — Performance drift
#
# Part 4 (final) of the ML Observability pipeline.
# Calculates performance metrics (WAPE, RMSE, MAE, F1) on production data
# and compares them against the baseline average to detect degradation.
# Drift is expressed as a proportional change.
# For error metrics (higher = worse): warn at +20%, critical at +50%.
# For F1 (lower = worse): warn at -15%, critical at -30%.
# Ends with a verification summary showing row counts across all landing tables.
#
# Prerequisite: 09a must have been executed to create landing tables.

# %% [markdown]
# ## 1. Setup

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F, Window, Row
session = get_active_session()

# %% [markdown]
# ### 1A. Constants

# %%
DATABASE = "BD_AA_DEV"
STORAGE_SCHEMA = "SC_FEATURES_BMX"

session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()

PREDICTION_TABLE = "DA_PREDICTIONS_VW"
ACTUALS_TABLE = "ACTUALS_TABLE_VW"
PERF_BASELINE = "DA_PERFORMANCE_BASELINE"

# Landing tables
DATA_DRIFT_HISTOGRAMS = "DA_DATA_DRIFT_HISTOGRAMS"
DATA_DRIFT = "DA_DATA_DRIFT"
PRED_DRIFT_HISTOGRAMS = "DA_PREDICTION_DRIFT_HISTOGRAMS"
PRED_DRIFT = "DA_PREDICTION_DRIFT"
PERF_DRIFT = "DA_PERFORMANCE"

MODEL_NAME = "UNI_BOX_REGRESSION_PARTITIONED"
AGG_COLS = ["STATS_NTILE_GROUP", "CUST_CATEGORY"]
TARGET_COL = "ACTUAL_SALES"
TIME_COL = "week"

PERF_JOIN_KEYS = ["CUSTOMER_ID", "BRAND_PRES_RET", TIME_COL]  # actuals table has no PROD_KEY
PERF_METRIC_NAMES = ["wape", "rmse", "mae", "f1_binary"]

# Thresholds for proportional drift (METRIC_DRIFT column)
# For error metrics (higher = worse): warn at +20%, critical at +50%
# For F1 (lower = worse): warn at -15%, critical at -30%
PERF_THRESHOLDS = {
    "wape":      {"warn": 0.20, "crit": 0.50, "direction": "higher_worse"},
    "rmse":      {"warn": 0.20, "crit": 0.50, "direction": "higher_worse"},
    "mae":       {"warn": 0.20, "crit": 0.50, "direction": "higher_worse"},
    "f1_binary": {"warn": -0.15, "crit": -0.30, "direction": "lower_worse"},
}

# %% [markdown]
# ### 1B. Functions

# %%
def get_new_combos(landing_table, agg_col, metric_col, min_week=None):
    """Find (WEEK, MODEL_VERSION) pairs in PREDICTION_TABLE not yet in landing_table."""
    all_combos = (
        session.table(PREDICTION_TABLE)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .select(
            F.col("ENTITY_MAP")["week"].cast("STRING").alias(TIME_COL),
            F.col("MODEL_VERSION"))
        .distinct()
    )
    if min_week is not None:
        all_combos = all_combos.filter(F.col(TIME_COL) >= min_week)

    existing_combos = (
        session.table(landing_table)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(F.col("AGGREGATED_COL") == agg_col.lower())
        .filter(F.col("METRIC_COL") == metric_col)
        .select(
            F.col("ENTITY_MAP")[TIME_COL].cast("STRING").alias(TIME_COL),
            F.col("MODEL_VERSION"),
        )
        .distinct()
    )
    new_combos = all_combos.join(
        existing_combos, on=[TIME_COL, "MODEL_VERSION"], how="left_anti"
    )
    new_combos_list = new_combos.collect()
    agg_label = f" [{agg_col}]" if agg_col else ""
    print(f"{landing_table}{agg_label}")
    print(f"  New ({TIME_COL}, MODEL_VERSION) combos: {len(new_combos_list)}")
    return new_combos_list

# %% [markdown]
# ## 4. Performance drift
#
# Calculates performance metrics (WAPE, RMSE, MAE, F1) on production data
# and compares them against the baseline average to detect degradation.

# %%
def compute_performance_metrics(paired_df, agg_col):
    """Compute WAPE, RMSE, MAE, and F1 from a paired predictions+actuals DataFrame."""
    agg_col_lower = agg_col.lower()
    actual = F.col(TARGET_COL)
    predicted = F.col("PREDICTION")
    abs_error = F.abs(actual - predicted)
    sq_error = F.pow(actual - predicted, F.lit(2))
    actual_positive = F.iff(actual > F.lit(0), F.lit(1), F.lit(0))
    predicted_positive = F.iff(predicted > F.lit(0), F.lit(1), F.lit(0))

    def _metrics_for_group(df, group_cols, agg_value_expr):
        """Compute the 4 metrics for a given grouping."""
        base = df.group_by(*group_cols).agg(
            F.sum(abs_error).alias("SUM_ABS_ERROR"),
            F.sum(F.abs(actual)).alias("SUM_ABS_ACTUAL"),
            F.avg(sq_error).alias("AVG_SQ_ERROR"),
            F.avg(abs_error).alias("MAE_VAL"),
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

    per_segment = _metrics_for_group(
        paired_df, [TIME_COL, "MODEL_VERSION", agg_col], F.col(agg_col).cast("STRING"),
    )
    full_model = _metrics_for_group(
        paired_df, [TIME_COL, "MODEL_VERSION"], F.lit("full_model"),
    )
    return per_segment.union_all_by_name(full_model)

# %%
# Get weeks where actuals are available (used to filter combos for performance metrics)
actuals_weeks = (
    session.table(ACTUALS_TABLE)
    .select(F.col(TIME_COL))
    .distinct()
    .collect()
)
actuals_week_set = {row[0] for row in actuals_weeks}
print(f"ACTUALS_TABLE has {len(actuals_week_set)} distinct weeks: {sorted(actuals_week_set)}")

# %%
# Get new (WEEK, MODEL_VERSION) combos for performance metrics, filtered to weeks with actuals

perf_new_combos_list = {}

for agg_col in AGG_COLS:
    all_combos = get_new_combos(
        PERF_DRIFT, agg_col=agg_col, metric_col=PERF_METRIC_NAMES[0])

    # Filter to only weeks where actuals exist
    perf_new_combos_list[agg_col] = [c for c in all_combos if c[0] in actuals_week_set]
    print(f"  After actuals filter: {len(perf_new_combos_list[agg_col])} combos for {agg_col}")

# %%
# Calculate inference performance metrics and drift — loop over each aggregation column

for agg_col in AGG_COLS:
    perf_combos = perf_new_combos_list[agg_col]

    if not perf_combos:
        print(f"  No new inference combos for {agg_col}, skipping.")
        continue

    # Read baseline averages for this agg_col
    baseline_avg = (
        session.table(PERF_BASELINE)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(F.col("AGGREGATED_COL") == agg_col.lower())
        .group_by("MODEL_VERSION", "AGGREGATED_COL", "AGGREGATED_VALUE", "METRIC_COL")
        .agg(F.avg("METRIC_VALUE").alias("BASELINE_METRIC"))
    )

    frames = []
    for combo in perf_combos:
        week, version = combo[0], combo[1]

        preds = (
            session.table(PREDICTION_TABLE)
            .filter(F.col("MODEL_NAME") == MODEL_NAME)
            .filter(F.col("MODEL_VERSION") == version)
            .with_column("CUSTOMER_ID", F.col("ENTITY_MAP")["customer_id"].cast("STRING"))
            .with_column("BRAND_PRES_RET", F.col("ENTITY_MAP")["brand_pres_ret"].cast("STRING"))
            .with_column("PROD_KEY", F.col("ENTITY_MAP")["prod_key"].cast("STRING"))
            .with_column(TIME_COL, F.col("ENTITY_MAP")["week"].cast("STRING"))  # TODO: fix once upstream data are corrected
            .filter(F.col(TIME_COL) == week)
        )
        actuals = (
            session.table(ACTUALS_TABLE)
            .filter(F.col(TIME_COL) == week)
            .drop(AGG_COLS)
        )

        # Left join: predictions drive the set; missing actuals coalesce to 0
        paired = (
            preds.join(actuals, on=PERF_JOIN_KEYS, how="left")
            .with_column(TARGET_COL, F.coalesce(F.col(TARGET_COL), F.lit(0.0)))
        )
        metrics_df = compute_performance_metrics(paired, agg_col)
        frames.append(metrics_df)

    all_metrics = frames[0]
    for f in frames[1:]:
        all_metrics = all_metrics.union_all_by_name(f)

    # Join with baseline averages to compute drift
    with_drift = all_metrics.join(
        baseline_avg,
        on=["MODEL_VERSION", "AGGREGATED_COL", "AGGREGATED_VALUE", "METRIC_COL"],
        how="left",
    ).with_column(
        "METRIC_DRIFT",
        (F.col("METRIC_VALUE") - F.col("BASELINE_METRIC"))
        / F.greatest(F.abs(F.col("BASELINE_METRIC")), F.lit(1e-10)),
    )

    data_date_expr = F.dateadd(
        TIME_COL, F.col(TIME_COL) % F.lit(100) - F.lit(1),
        F.date_from_parts(F.floor(F.col(TIME_COL) / F.lit(100)), F.lit(1), F.lit(1)),
    )

    warn_expr = (
        F.when(F.col("METRIC_COL") == "f1_binary", F.lit(PERF_THRESHOLDS["f1_binary"]["warn"]))
         .otherwise(F.lit(PERF_THRESHOLDS["wape"]["warn"]))
         .cast("DOUBLE")
    )
    crit_expr = (
        F.when(F.col("METRIC_COL") == "f1_binary", F.lit(PERF_THRESHOLDS["f1_binary"]["crit"]))
         .otherwise(F.lit(PERF_THRESHOLDS["wape"]["crit"]))
         .cast("DOUBLE")
    )

    alert_expr = (
        F.when(
            F.col("METRIC_COL") == "f1_binary",
            F.when(F.col("METRIC_DRIFT") < F.lit(PERF_THRESHOLDS["f1_binary"]["crit"]), F.lit(2))
             .when(F.col("METRIC_DRIFT") < F.lit(PERF_THRESHOLDS["f1_binary"]["warn"]), F.lit(1))
             .otherwise(F.lit(0))
        ).otherwise(
            F.when(F.col("METRIC_DRIFT") > F.lit(PERF_THRESHOLDS["wape"]["crit"]), F.lit(2))
             .when(F.col("METRIC_DRIFT") > F.lit(PERF_THRESHOLDS["wape"]["warn"]), F.lit(1))
             .otherwise(F.lit(0))
        )
    )

    drift_rows = (
        with_drift
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
            F.col("METRIC_VALUE").cast("DOUBLE"),
            F.col("METRIC_DRIFT").cast("DOUBLE"),
            warn_expr.alias("WARNING_THRESHOLD"),
            crit_expr.alias("CRITICAL_THRESHOLD"),
            alert_expr.alias("ALERT_LEVEL"),
            F.lit("MXBEB").alias("BKCC"),
            F.to_char(F.col("DATA_DATE"), F.lit("YYYYMM")).alias("CALMONTH"),
            F.current_timestamp().alias("LDTS"),
        )
    )

    drift_rows.write.mode("append").save_as_table(PERF_DRIFT)

    n_rows = session.table(PERF_DRIFT).count()
    print(f"Perf drift for {agg_col}: PERF_DRIFT now has {n_rows} rows.")

# %% [markdown]
# ## 5. Verification summary
# #
# Final summary showing the row count in each landing table to verify
# that all metrics were calculated correctly.

# %%
print("=" * 60)
print("ML Observability — Landing Table Summary")
print("=" * 60)
for tbl_name, tbl_var in [
    ("DA_DATA_DRIFT_HISTOGRAMS", DATA_DRIFT_HISTOGRAMS),
    ("DA_DATA_DRIFT", DATA_DRIFT),
    ("DA_PREDICTION_DRIFT_HISTOGRAMS", PRED_DRIFT_HISTOGRAMS),
    ("DA_PREDICTION_DRIFT", PRED_DRIFT),
    ("DA_PERFORMANCE", PERF_DRIFT),
]:
    n = session.table(tbl_var).count()
    print(f"  {tbl_name:.<45} {n:>8} rows")
print("=" * 60)
