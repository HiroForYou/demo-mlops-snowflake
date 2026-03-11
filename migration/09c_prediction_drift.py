# %% [markdown]
# # 09c — Prediction drift
#
# Part 3 of the ML Observability pipeline.
# Detects changes in the model prediction distribution relative to the baseline.
# Uses Jensen-Shannon methodology applied to the prediction column.
# Histograms use baseline bin edges then JSD is calculated.
# Thresholds: >0.2 = warning, >0.45 = critical.
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
PRED_DRIFT_HISTOGRAMS_BASELINE = "DA_PREDICTION_DRIFT_HISTOGRAMS_BASELINE"
PRED_DRIFT_HISTOGRAMS = "DA_PREDICTION_DRIFT_HISTOGRAMS"
PRED_DRIFT = "DA_PREDICTION_DRIFT"

MODEL_NAME = "UNI_BOX_REGRESSION_PARTITIONED"
AGG_COLS = ["STATS_NTILE_GROUP", "CUST_CATEGORY"]
PREDICTION_COL = "PREDICTED_UNI_BOX_WEEK"
TIME_COL = "week"

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
# ## 3. Prediction drift

# %%
def get_prediction_bin_edges(model_version, agg_col):
    """Retrieve stored baseline bin edges for prediction drift."""
    bins = (
        session.table(PRED_DRIFT_HISTOGRAMS_BASELINE)
        .filter(
            (F.col("MODEL_VERSION") == model_version)
            & (F.col("MODEL_NAME") == MODEL_NAME)
            & (F.col("AGGREGATED_COL") == agg_col.lower())
            & (F.col("METRIC_COL") == "jensen-shannon")
        )
        .select(
            F.col("AGGREGATED_VALUE"),
            F.col("METRIC_MAP")["bin_number"].cast("INT").alias("BIN"),
            F.col("METRIC_MAP")["bin_low"].cast("FLOAT").alias("BIN_LOW"),
            F.col("METRIC_MAP")["bin_high"].cast("FLOAT").alias("BIN_HIGH"),
        )
        .distinct()
    )
    return bins

# %%
def compute_prediction_drift_histograms(combos, target_table, agg_col):
    """Compute inference prediction histograms per segment using baseline bin edges."""
    agg_col_lower = agg_col.lower()
    frames = []

    for combo in combos:
        week, version = combo[0], combo[1]
        src_tbl = (
            session.table(PREDICTION_TABLE)
            .with_column(TIME_COL, F.col("ENTITY_MAP")["week"])  # TODO: fix once upstream data are corrected
            .filter(F.col("MODEL_NAME") == MODEL_NAME)
            .filter(F.col(TIME_COL) == week)
        )
        bins = get_prediction_bin_edges(version, agg_col)

        pred_values = (
            src_tbl
            .select(
                F.col(TIME_COL),
                F.col(agg_col).cast("STRING").alias("AGGREGATED_VALUE"),
                F.col("PREDICTION"),
            )
            .filter(F.col("PREDICTION").is_not_null())
        )

        actual_counts = (
            pred_values
            .join(bins, on=["AGGREGATED_VALUE"])
            .filter(
                (F.col("PREDICTION") >= F.col("BIN_LOW"))
                & (F.col("PREDICTION") < F.col("BIN_HIGH"))
            )
            .group_by(TIME_COL, "AGGREGATED_VALUE", "BIN", "BIN_LOW", "BIN_HIGH")
            .agg(F.count("*").alias("BIN_COUNT"))
        )

        distinct_times = src_tbl.select(F.col(TIME_COL)).distinct()
        scaffold = distinct_times.cross_join(bins)

        binned = (
            scaffold.join(
                actual_counts,
                on=[TIME_COL, "AGGREGATED_VALUE", "BIN", "BIN_LOW", "BIN_HIGH"],
                how="left",
            )
            .with_column("BIN_COUNT", F.coalesce(F.col("BIN_COUNT"), F.lit(0)))
        )

        data_date_expr = F.dateadd(
            "week",
            F.col(TIME_COL) % F.lit(100) - F.lit(1),
            F.date_from_parts(F.floor(F.col(TIME_COL) / F.lit(100)), F.lit(1), F.lit(1)),
        )

        histograms = (
            binned
            .with_column("DATA_DATE", data_date_expr)
            .select(
                F.sha2(
                    F.concat(
                        F.lit(MODEL_NAME), F.lit("||"),
                        F.lit(version), F.lit("||"),
                        F.lit(PREDICTION_COL), F.lit("||"),
                        F.col(TIME_COL), F.lit("||"),
                        F.lit(agg_col_lower), F.lit("||"),
                        F.col("AGGREGATED_VALUE").cast("STRING"), F.lit("||"),
                        F.lit("jensen-shannon"), F.lit("||"),
                        F.col("BIN").cast("STRING"),
                    ),
                    256,
                ).alias("RECORD_ID"),
                F.lit(MODEL_NAME).alias("MODEL_NAME"),
                F.lit(version).alias("MODEL_VERSION"),
                F.object_construct(
                    F.lit("prediction_name"), F.lit(PREDICTION_COL),
                    F.lit(TIME_COL), F.col(TIME_COL),
                    F.lit("data_date"), F.col("DATA_DATE"),
                ).alias("ENTITY_MAP"),
                F.lit(agg_col_lower).alias("AGGREGATED_COL"),
                F.col("AGGREGATED_VALUE"),
                F.lit("jensen-shannon").alias("METRIC_COL"),
                F.object_construct(
                    F.lit("bin_number"), F.col("BIN"),
                    F.lit("bin_low"), F.col("BIN_LOW"),
                    F.lit("bin_high"), F.col("BIN_HIGH"),
                    F.lit("bin_count"), F.col("BIN_COUNT"),
                ).alias("METRIC_MAP"),
                F.to_char(F.col("DATA_DATE"), F.lit("YYYYMM")).alias("CALMONTH"),
                F.current_timestamp().alias("LDTS"),
            )
        )
        frames.append(histograms)

    result = frames[0]
    for f in frames[1:]:
        result = result.union_all_by_name(f)

    result.write.mode("append").save_as_table(target_table)
    n_rows = session.table(target_table).count()
    print(f"Wrote {len(combos)} combo(s) for {agg_col}. {target_table} now has {n_rows} rows.")

# %%
# Get new (WEEK, MODEL_VERSION) combos for prediction histograms

pred_new_combos_list = {}

for agg_col in AGG_COLS:
    pred_new_combos_list[agg_col] = get_new_combos(
        PRED_DRIFT_HISTOGRAMS, agg_col=agg_col,
        metric_col="jensen-shannon")

# %%
# Calculate prediction histograms — loop over each aggregation column

for agg_col in AGG_COLS:
    if pred_new_combos_list[agg_col]:
        compute_prediction_drift_histograms(
            pred_new_combos_list[agg_col],
            target_table=PRED_DRIFT_HISTOGRAMS, agg_col=agg_col)
    else:
        print(f"  No new combos for {agg_col} prediction histograms, skipping.")

# %%
EPSILON = F.lit(1e-10)

# Compute Jensen-Shannon Distance from prediction histograms
for agg_col in AGG_COLS:
    agg_col_lower = agg_col.lower()

    if not pred_new_combos_list[agg_col]:
        print(f"  No new combos for {agg_col} prediction JSD, skipping.")
        continue

    combo_keys = [F.lit(f'{row[0]}|{row[1]}') for row in pred_new_combos_list[agg_col]]

    jsd_common_filters = [
        F.col("METRIC_COL") == "jensen-shannon",
        F.col("AGGREGATED_COL") == agg_col_lower,
    ]

    def _add_jsd_cols(df):
        return (df
            .with_column(TIME_COL, F.col("ENTITY_MAP")[TIME_COL].cast("STRING"))
            .with_column("BIN", F.col("METRIC_MAP")["bin_number"].cast("INT"))
            .with_column("BIN_COUNT", F.col("METRIC_MAP")["bin_count"].cast("INT"))
        )

    baseline_hist = _add_jsd_cols(
        session.table(PRED_DRIFT_HISTOGRAMS_BASELINE)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(jsd_common_filters[0]).filter(jsd_common_filters[1])
    )
    inference_hist = _add_jsd_cols(
        session.table(PRED_DRIFT_HISTOGRAMS)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(jsd_common_filters[0]).filter(jsd_common_filters[1])
    )

    baseline_total = F.sum("BIN_COUNT").over(
        Window.partition_by(TIME_COL, "AGGREGATED_VALUE")
    )
    baseline = (
        baseline_hist
        .with_column("BIN_PROB", F.col("BIN_COUNT") / F.greatest(baseline_total, F.lit(1.0)))
        .group_by("AGGREGATED_VALUE", "BIN")
        .agg(F.avg("BIN_PROB").alias("P"))
    )

    inf_total = F.sum("BIN_COUNT").over(
        Window.partition_by(TIME_COL, "MODEL_VERSION", "AGGREGATED_VALUE")
    )
    inference = (
        inference_hist
        .filter(
            F.concat(F.col(TIME_COL), F.lit("|"), F.col("MODEL_VERSION"))
            .isin(combo_keys)
        )
        .with_column("Q", F.col("BIN_COUNT") / inf_total)
        .select(TIME_COL, "MODEL_VERSION", "AGGREGATED_VALUE", "BIN", F.col("Q"))
    )

    joined = inference.join(baseline, on=["AGGREGATED_VALUE", "BIN"], how="full")

    p = F.greatest(F.coalesce(F.col("P"), F.lit(0.0)), EPSILON)
    q = F.greatest(F.coalesce(F.col("Q"), F.lit(0.0)), EPSILON)
    m = (p + q) / F.lit(2.0)

    kl_p_m = p * F.ln(p / m)
    kl_q_m = q * F.ln(q / m)

    jsd_df = (
        joined
        .with_column("KL_P_M", kl_p_m)
        .with_column("KL_Q_M", kl_q_m)
        .group_by(TIME_COL, "MODEL_VERSION", "AGGREGATED_VALUE")
        .agg(
            F.sum("KL_P_M").alias("SUM_KL_P_M"),
            F.sum("KL_Q_M").alias("SUM_KL_Q_M"),
        )
        .with_column(
            "JS_DISTANCE",
            F.sqrt(F.lit(0.5) * F.col("SUM_KL_P_M") + F.lit(0.5) * F.col("SUM_KL_Q_M")),
        )
        .with_column(
            "DATA_DATE",
            F.dateadd("week", F.col(TIME_COL) % F.lit(100) - F.lit(1),
                      F.date_from_parts(F.floor(F.col(TIME_COL) / F.lit(100)), F.lit(1), F.lit(1))),
        )
        .select(
            F.sha2(
                F.concat(
                    F.lit(MODEL_NAME), F.lit("||"),
                    F.col("MODEL_VERSION"), F.lit("||"),
                    F.col(TIME_COL), F.lit("||"),
                    F.lit(agg_col_lower), F.lit("||"),
                    F.col("AGGREGATED_VALUE").cast("STRING"),
                    F.lit("jensen-shannon"),
                ),
                256,
            ).alias("RECORD_ID"),
            F.lit(MODEL_NAME).alias("MODEL_NAME"),
            F.col("MODEL_VERSION"),
            F.object_construct(
                F.lit("prediction_name"), F.lit(PREDICTION_COL),
                F.lit(TIME_COL), F.col(TIME_COL),
                F.lit("data_date"), F.col("DATA_DATE"),
            ).alias("ENTITY_MAP"),
            F.lit(agg_col_lower).alias("AGGREGATED_COL"),
            F.col("AGGREGATED_VALUE").cast("STRING").alias("AGGREGATED_VALUE"),
            F.lit("jensen-shannon").alias("METRIC_COL"),
            F.col("JS_DISTANCE").cast("DOUBLE").alias("METRIC_VALUE"),
            F.lit(0.2).cast("DOUBLE").alias("WARNING_THRESHOLD"),
            F.lit(0.45).cast("DOUBLE").alias("CRITICAL_THRESHOLD"),
            F.when(F.col("JS_DISTANCE") > 0.45, F.lit(2))
             .when(F.col("JS_DISTANCE") > 0.2, F.lit(1))
             .otherwise(F.lit(0)).alias("ALERT_LEVEL"),
            F.lit("MXBEB").alias("BKCC"),
            F.to_char(F.col("DATA_DATE"), F.lit("YYYYMM")).alias("CALMONTH"),
            F.current_timestamp().alias("LDTS"),
        )
    )

    jsd_df.write.mode("append").save_as_table(PRED_DRIFT)

    n_rows = session.table(PRED_DRIFT).count()
    print(f"Prediction JSD for {agg_col}: PRED_DRIFT now has {n_rows} rows.")
