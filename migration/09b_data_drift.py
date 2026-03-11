# %% [markdown]
# # 09b — Data drift
#
# Part 2 of the ML Observability pipeline.
# Detects changes in the input data distribution relative to the baseline.
# Two complementary metrics are calculated:
# - PSI (segment composition stability)
# - Jensen-Shannon Distance (individual numerical feature drift)
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
# Account info
DATABASE = "BD_AA_DEV"
STORAGE_SCHEMA = "SC_FEATURES_BMX"
FEATURES_SCHEMA = "SC_FEATURES_BMX"

session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()

# Input data sources
FEATURE_TABLE = f"{DATABASE}.{FEATURES_SCHEMA}.INFERENCE_DATASET_CLEANED_VW"
PREDICTION_TABLE = "DA_PREDICTIONS_VW"

# Baseline tables (read-only)
DATA_DRIFT_HISTOGRAMS_BASELINE = f"DA_DATA_DRIFT_HISTOGRAMS_BASELINE"

# Landing tables
DATA_DRIFT_HISTOGRAMS = f"DA_DATA_DRIFT_HISTOGRAMS"
DATA_DRIFT = f"DA_DATA_DRIFT"

# Model specifics
MODEL_NAME = "UNI_BOX_REGRESSION_PARTITIONED"
AGG_COLS = ["STATS_NTILE_GROUP", "CUST_CATEGORY"]
TIME_COL = "week"

NON_FEATURE_COLS = {
    "customer_id", "brand_pres_ret", "prod_key",
    TIME_COL, "ACTUAL_SALES", "STATS_NTILE_GROUP", "CUST_CATEGORY",
}

NON_DRIFT_COLS = {'WEEK_OF_YEAR'}

# Feature drift settings
N_BINS = 20

# %% [markdown]
# ### 1B. Functions

# %%
def get_new_combos(landing_table, agg_col, metric_col, min_week=None):
    """Find (WEEK, MODEL_VERSION) pairs in PREDICTION_TABLE not yet in landing_table.

    Parameters
    ----------
    landing_table : str
        Fully-qualified landing table name.
    agg_col : str
        Aggregation column being selected.
    metric_col : str
        Metric being selected.
    min_week : str or None
        If provided, only consider inference weeks >= this value.

    Returns
    -------
    list[Row]
        (WEEK, MODEL_VERSION) pairs needing computation.
    """
    all_combos = (
        session.table(PREDICTION_TABLE)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .select(
            F.col("ENTITY_MAP")["week"].cast("STRING").alias(TIME_COL), # TODO: fix once upstream data are corrected
            F.col("MODEL_VERSION"))
        .distinct()
    )
    if min_week is not None:
        all_combos = all_combos.filter(F.col(TIME_COL) >= min_week)

    # Check landing table for existing combos
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
# ## 2. Data drift
#
# Detects changes in the input data distribution relative to the baseline.
# Two complementary metrics are calculated: PSI (segment composition stability)
# and Jensen-Shannon Distance (individual numerical feature drift).

# %% [markdown]
# ### 2A. Segment stability (PSI)
#
# Calculates the Population Stability Index (PSI) by comparing customer proportions
# per segment in production vs. the baseline. It first generates segment histograms
# and then calculates the aggregate PSI. Thresholds: >0.1 = warning, >0.2 = critical.

# %%
def compute_segment_histograms(combos, target_table, agg_col):
    """Compute per-segment customer proportions from inference data and write to target_table.

    Parameters
    ----------
    combos : list[Row]
        (WEEK, MODEL_VERSION) pairs to compute.
    target_table : str
        Table to write results to.
    agg_col : str
        Column to aggregate by (e.g. "STATS_NTILE_GROUP").
    """
    agg_col_lower = agg_col.lower()
    frames = []

    for combo in combos:
        week, version = combo[0], combo[1]
        src_tbl = session.table(FEATURE_TABLE).filter(F.col(TIME_COL) == week)

        data_date_expr = F.dateadd(
            "week", F.col(TIME_COL) % F.lit(100) - F.lit(1),
            F.date_from_parts(F.floor(F.col(TIME_COL) / F.lit(100)), F.lit(1), F.lit(1)),
        )

        proportions = (
            src_tbl
            .group_by(TIME_COL, agg_col)
            # TODO: review whether count_distinct(CUSTOMER_ID) is more appropriate than row count
            .agg(F.count(F.lit(1)).alias("N_ROWS"))
            .with_column(
                "PROPORTION",
                F.col("N_ROWS")
                / F.sum("N_ROWS").over(Window.partition_by(TIME_COL)),
            )
            .with_column("DATA_DATE", data_date_expr)
            .select(
                F.sha2(
                    F.concat(
                        F.lit(MODEL_NAME), F.lit("||"),
                        F.lit(version), F.lit("||"),
                        F.col(TIME_COL), F.lit("||"),
                        F.lit(agg_col_lower), F.lit("||"),
                        F.col(agg_col).cast("STRING"), F.lit("||"),
                        F.lit("population_stability_index"),
                    ),
                    256,
                ).alias("RECORD_ID"),
                F.lit(MODEL_NAME).alias("MODEL_NAME"),
                F.lit(version).alias("MODEL_VERSION"),
                F.object_construct(
                    F.lit(TIME_COL), F.col(TIME_COL),
                    F.lit("data_date"), F.col("DATA_DATE"),
                ).alias("ENTITY_MAP"),
                F.lit(agg_col_lower).alias("AGGREGATED_COL"),
                F.col(agg_col).cast("STRING").alias("AGGREGATED_VALUE"),
                F.lit("population_stability_index").alias("METRIC_COL"),
                F.object_construct(
                    F.lit("bin_count"), F.col("N_ROWS"),
                    F.lit("bin_prop"), F.col("PROPORTION"),
                ).alias("METRIC_MAP"),
                F.to_char(F.col("DATA_DATE"), F.lit("YYYYMM")).alias("CALMONTH"),
                F.current_timestamp().alias("LDTS"),
            )
        )
        frames.append(proportions)

    result = frames[0]
    for f in frames[1:]:
        result = result.union_all_by_name(f)

    result.write.mode("append").save_as_table(target_table)

    n_rows = session.table(target_table).count()
    print(f"Wrote {len(combos)} combo(s) for {agg_col}. {target_table} now has {n_rows} rows.")

# %%
# Get new (WEEK, MODEL_VERSION) combos for segment histograms

psi_new_combos_list = {}

for agg_col in AGG_COLS:
    psi_new_combos_list[agg_col] = get_new_combos(
        DATA_DRIFT_HISTOGRAMS, agg_col=agg_col,
        metric_col="population_stability_index")

# %%
# Calculate segment proportions — loop over each aggregation column

for agg_col in AGG_COLS:
    if psi_new_combos_list[agg_col]:
        compute_segment_histograms(
            psi_new_combos_list[agg_col],
            target_table=DATA_DRIFT_HISTOGRAMS, agg_col=agg_col)
    else:
        print(f"  No new combos for {agg_col} segment histograms, skipping.")

# %%
# Compute PSI from histograms

EPSILON = F.lit(1e-10)

for agg_col in AGG_COLS:
    if not psi_new_combos_list[agg_col]:
        print(f"  No new combos for {agg_col} PSI, skipping.")
        continue

    combo_keys = [F.lit(f'{row[0]}|{row[1]}') for row in psi_new_combos_list[agg_col]]

    # Baseline: average proportion per segment
    baseline = (
        session.table(DATA_DRIFT_HISTOGRAMS_BASELINE)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(F.col("METRIC_COL") == "population_stability_index")
        .filter(F.col("AGGREGATED_COL") == agg_col.lower())
        .select(
            F.col("AGGREGATED_VALUE"),
            F.col("METRIC_MAP")["bin_prop"].cast("DOUBLE").alias("PROPORTION"),
        )
        .group_by("AGGREGATED_VALUE")
        .agg(F.avg("PROPORTION").alias("BASELINE_PROPORTION"))
    )

    # Inference: only new combos
    inference = (
        session.table(DATA_DRIFT_HISTOGRAMS)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(F.col("METRIC_COL") == "population_stability_index")
        .filter(F.col("AGGREGATED_COL") == agg_col.lower())
        .with_column(TIME_COL, F.col("ENTITY_MAP")[TIME_COL].cast("STRING"))
        .filter(
            F.concat(F.col(TIME_COL), F.lit("|"), F.col("MODEL_VERSION"))
            .isin(combo_keys)
        )
        .select(F.col(TIME_COL), F.col("MODEL_VERSION"),
                F.col("AGGREGATED_VALUE"),
                F.col("METRIC_MAP")["bin_prop"].cast("DOUBLE").alias("INFERENCE_PROPORTION"))
    )

    # PSI per (WEEK, MODEL_VERSION)
    inf_prop = F.greatest(F.col("INFERENCE_PROPORTION"), EPSILON)
    base_prop = F.greatest(F.col("BASELINE_PROPORTION"), EPSILON)
    psi_component = (inf_prop - base_prop) * F.ln(inf_prop / base_prop)

    psi_df = (
        inference.join(baseline, on="AGGREGATED_VALUE")
        .with_column("PSI_COMPONENT", psi_component)
        .group_by(TIME_COL, "MODEL_VERSION")
        .agg(F.sum("PSI_COMPONENT").alias("PSI"))
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
                    F.lit(agg_col.lower()), F.lit("||"),
                    F.lit("full_model"), F.lit("||"),
                    F.lit("population_stability_index"),
                ),
                256,
            ).alias("RECORD_ID"),
            F.lit(MODEL_NAME).alias("MODEL_NAME"),
            F.col("MODEL_VERSION"),
            F.object_construct(
                F.lit(TIME_COL), F.col(TIME_COL),
                F.lit("data_date"), F.col("DATA_DATE"),
            ).alias("ENTITY_MAP"),
            F.lit(agg_col.lower()).alias("AGGREGATED_COL"),
            F.lit("full_model").alias("AGGREGATED_VALUE"),
            F.lit("population_stability_index").alias("METRIC_COL"),
            F.col("PSI").cast("DOUBLE").alias("METRIC_VALUE"),
            F.lit(0.1).cast("DOUBLE").alias("WARNING_THRESHOLD"),
            F.lit(0.2).cast("DOUBLE").alias("CRITICAL_THRESHOLD"),
            F.when(F.col("PSI") > 0.2, F.lit(2))
             .when(F.col("PSI") > 0.1, F.lit(1))
             .otherwise(F.lit(0)).alias("ALERT_LEVEL"),
            F.lit("MXBEB").alias("BKCC"),
            F.to_char(F.col("DATA_DATE"), F.lit("YYYYMM")).alias("CALMONTH"),
            F.current_timestamp().alias("LDTS"),
        )
    )

    psi_df.write.mode("append").save_as_table(DATA_DRIFT)

    n_rows = session.table(DATA_DRIFT).count()
    print(f"PSI for {agg_col}: DATA_DRIFT now has {n_rows} rows.")

# %% [markdown]
# ### 2B. Feature drift (Jensen-Shannon distance)
#
# Calculates the Jensen-Shannon distance per numerical feature, comparing production
# histograms against baseline ones. Reuses bin edges stored in the baseline
# to ensure consistency. Thresholds: >0.2 = warning, >0.45 = critical.

# %%
def get_feature_bin_edges(model_version, agg_col):
    """Retrieve stored baseline bin edges for feature drift.

    Reads edges from DATA_DRIFT_HISTOGRAMS_BASELINE filtered by model_version.

    Parameters
    ----------
    model_version : str
        Model version string to filter baseline rows.
    agg_col : str
        Column name to aggregate by (e.g. "STATS_NTILE_GROUP").

    Returns
    -------
    tuple[snowpark.DataFrame, list[str]]
        (bins_df, features) where bins_df has columns
        AGGREGATED_VALUE, FEATURE_NAME, BIN, BIN_LOW, BIN_HIGH;
        and features is the list of monitored feature names.
    """
    bins = (
        session.table(DATA_DRIFT_HISTOGRAMS_BASELINE)
        .filter(
            (F.col("MODEL_VERSION") == model_version)
            & (F.col("MODEL_NAME") == MODEL_NAME)
            & (F.col("AGGREGATED_COL") == agg_col.lower())
            & (F.col("METRIC_COL") == "jensen-shannon")
        )
        .select(
            F.col("AGGREGATED_VALUE"),
            F.col("ENTITY_MAP")["feature_name"].cast("STRING").alias("FEATURE_NAME"),
            F.col("METRIC_MAP")["bin_number"].cast("INT").alias("BIN"),
            F.col("METRIC_MAP")["bin_low"].cast("FLOAT").alias("BIN_LOW"),
            F.col("METRIC_MAP")["bin_high"].cast("FLOAT").alias("BIN_HIGH"),
        )
        .distinct()
    )
    features = [
        row["FEATURE_NAME"]
        for row in bins.select("FEATURE_NAME").distinct().collect()
    ]

    return bins, features

# %%
def compute_data_drift_histograms(combos, target_table, agg_col):
    """Compute per-feature, per-segment inference histograms using baseline bin edges.

    Parameters
    ----------
    combos : list[Row]
        (WEEK, MODEL_VERSION) pairs to compute.
    target_table : str
        Table to write results to.
    agg_col : str
        Column name to aggregate by (e.g. "STATS_NTILE_GROUP").
    """
    agg_col_lower = agg_col.lower()
    frames = []

    for combo in combos:
        week, version = combo[0], combo[1]
        src_tbl = session.table(FEATURE_TABLE).filter(F.col(TIME_COL) == week)

        # Get bin edges from stored baseline
        bins, features = get_feature_bin_edges(version, agg_col)

        # Unpivot source data into (TIME, SEGMENT, FEATURE_NAME, FEATURE_VALUE)
        cast_exprs = [F.col(f).cast("FLOAT").alias(f) for f in features]
        src_casted = src_tbl.select(
            F.col(TIME_COL), F.col(agg_col), *cast_exprs
        )
        unpivoted = src_casted.unpivot(
            "FEATURE_VALUE", "FEATURE_NAME",
            [F.col(feat) for feat in features],
        ).select(
            F.col(TIME_COL),
            F.col(agg_col).cast("STRING").alias("AGGREGATED_VALUE"),
            "FEATURE_NAME", "FEATURE_VALUE",
        ).filter(F.col("FEATURE_VALUE").is_not_null())

        # Count values per bin
        actual_counts = (
            unpivoted
            .join(bins, on=["FEATURE_NAME", "AGGREGATED_VALUE"])
            .filter(
                (F.col("FEATURE_VALUE") >= F.col("BIN_LOW"))
                & (F.col("FEATURE_VALUE") < F.col("BIN_HIGH"))
            )
            .group_by(TIME_COL, "AGGREGATED_VALUE", "FEATURE_NAME", "BIN", "BIN_LOW", "BIN_HIGH")
            .agg(F.count("*").alias("BIN_COUNT"))
        )

        # Scaffold: every (TIME x bin) so empty bins get a row
        distinct_times = src_casted.select(F.col(TIME_COL)).distinct()
        scaffold = distinct_times.cross_join(bins)

        binned = (
            scaffold.join(
                actual_counts,
                on=[TIME_COL, "AGGREGATED_VALUE", "FEATURE_NAME", "BIN", "BIN_LOW", "BIN_HIGH"],
                how="left",
            )
            .with_column("BIN_COUNT", F.coalesce(F.col("BIN_COUNT"), F.lit(0)))
        )

        # Compute DATA_DATE from WEEK
        data_date_expr = F.dateadd(
            "week",
            F.col(TIME_COL) % F.lit(100) - F.lit(1),
            F.date_from_parts(
                F.floor(F.col(TIME_COL) / F.lit(100)),
                F.lit(1), F.lit(1),
            ),
        )

        histograms = (
            binned
            .with_column("DATA_DATE", data_date_expr)
            .select(
                F.sha2(
                    F.concat(
                        F.lit(MODEL_NAME), F.lit("||"),
                        F.lit(version), F.lit("||"),
                        F.col(TIME_COL), F.lit("||"),
                        F.lit(agg_col_lower), F.lit("||"),
                        F.col("AGGREGATED_VALUE").cast("STRING"), F.lit("||"),
                        F.col("FEATURE_NAME"), F.lit("||"),
                        F.col("BIN").cast("STRING"), F.lit("||"),
                        F.lit("jensen-shannon"),
                    ),
                    256,
                ).alias("RECORD_ID"),
                F.lit(MODEL_NAME).alias("MODEL_NAME"),
                F.lit(version).alias("MODEL_VERSION"),
                F.object_construct(
                    F.lit("feature_name"), F.col("FEATURE_NAME"),
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
# Get new (WEEK, MODEL_VERSION) combos for feature histograms

jsd_new_combos_list = {}

for agg_col in AGG_COLS:
    jsd_new_combos_list[agg_col] = get_new_combos(
        DATA_DRIFT_HISTOGRAMS, agg_col=agg_col,
        metric_col="jensen-shannon")

# %%
# Calculate feature histograms — loop over each aggregation column

for agg_col in AGG_COLS:
    if jsd_new_combos_list[agg_col]:
        compute_data_drift_histograms(
            jsd_new_combos_list[agg_col],
            target_table=DATA_DRIFT_HISTOGRAMS, agg_col=agg_col)
    else:
        print(f"  No new combos for {agg_col} feature histograms, skipping.")

# %%
EPSILON = F.lit(1e-10)

# Compute Jensen-Shannon Distance from feature histograms
for agg_col in AGG_COLS:
    agg_col_lower = agg_col.lower()

    if not jsd_new_combos_list[agg_col]:
        print(f"  No new combos for {agg_col} JSD, skipping.")
        continue

    combo_keys = [F.lit(f'{row[0]}|{row[1]}') for row in jsd_new_combos_list[agg_col]]

    jsd_common_filters = [
        F.col("METRIC_COL") == "jensen-shannon",
        F.col("AGGREGATED_COL") == agg_col_lower,
    ]

    def _add_jsd_cols(df):
        return (df
            .with_column(TIME_COL, F.col("ENTITY_MAP")[TIME_COL].cast("STRING"))
            .with_column("FEATURE_NAME", F.col("ENTITY_MAP")["feature_name"].cast("STRING"))
            .with_column("BIN", F.col("METRIC_MAP")["bin_number"].cast("INT"))
            .with_column("BIN_COUNT", F.col("METRIC_MAP")["bin_count"].cast("INT"))
        )

    baseline_hist = _add_jsd_cols(
        session.table(DATA_DRIFT_HISTOGRAMS_BASELINE)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(jsd_common_filters[0]).filter(jsd_common_filters[1])
    )
    inference_hist = _add_jsd_cols(
        session.table(DATA_DRIFT_HISTOGRAMS)
        .filter(F.col("MODEL_NAME") == MODEL_NAME)
        .filter(jsd_common_filters[0]).filter(jsd_common_filters[1])
    )

    # Baseline: average bin probability per (SEGMENT, FEATURE_NAME, BIN)
    baseline_total = F.sum("BIN_COUNT").over(
        Window.partition_by(TIME_COL, "AGGREGATED_VALUE", "FEATURE_NAME")
    )
    baseline = (
        baseline_hist
        .with_column("BIN_PROB", F.col("BIN_COUNT") / F.greatest(baseline_total, F.lit(1.0)))
        .group_by("AGGREGATED_VALUE", "FEATURE_NAME", "BIN")
        .agg(F.avg("BIN_PROB").alias("P"))
    )

    # Inference: probabilities for new combos only
    inf_total = F.sum("BIN_COUNT").over(
        Window.partition_by(TIME_COL, "MODEL_VERSION", "AGGREGATED_VALUE", "FEATURE_NAME")
    )
    inference = (
        inference_hist
        .filter(
            F.concat(F.col(TIME_COL), F.lit("|"), F.col("MODEL_VERSION"))
            .isin(combo_keys)
        )
        .with_column("Q", F.col("BIN_COUNT") / inf_total)
        .select(TIME_COL, "MODEL_VERSION", "AGGREGATED_VALUE", "FEATURE_NAME", "BIN",
                F.col("Q"))
    )

    # Full outer join so bins in only one distribution contribute
    joined = inference.join(
        baseline,
        on=["AGGREGATED_VALUE", "FEATURE_NAME", "BIN"],
        how="full",
    )

    p = F.greatest(F.coalesce(F.col("P"), F.lit(0.0)), EPSILON)
    q = F.greatest(F.coalesce(F.col("Q"), F.lit(0.0)), EPSILON)
    m = (p + q) / F.lit(2.0)

    kl_p_m = p * F.ln(p / m)
    kl_q_m = q * F.ln(q / m)

    # JSD = sqrt(0.5 * KL(P||M) + 0.5 * KL(Q||M))
    jsd_df = (
        joined
        .with_column("KL_P_M", kl_p_m)
        .with_column("KL_Q_M", kl_q_m)
        .group_by(TIME_COL, "MODEL_VERSION", "AGGREGATED_VALUE", "FEATURE_NAME")
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
                    F.col("AGGREGATED_VALUE").cast("STRING"), F.lit("||"),
                    F.col("FEATURE_NAME"), F.lit("||"),
                    F.lit("jensen-shannon"),
                ),
                256,
            ).alias("RECORD_ID"),
            F.lit(MODEL_NAME).alias("MODEL_NAME"),
            F.col("MODEL_VERSION"),
            F.object_construct(
                F.lit("feature_name"), F.col("FEATURE_NAME"),
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

    jsd_df.write.mode("append").save_as_table(DATA_DRIFT)

    n_rows = session.table(DATA_DRIFT).count()
    print(f"JSD for {agg_col}: DATA_DRIFT now has {n_rows} rows.")
