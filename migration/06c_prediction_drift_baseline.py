# %% [markdown]
# # 06c — Prediction drift baseline
#
# Part 3 of the baseline generation pipeline.
# Calculates baseline histograms of the prediction distribution per segment.
# The same quantile-based bins as for features are used, but applied
# to the prediction column. These histograms serve as a reference to
# detect drift in the model predictions in production.
#
# Prerequisite: 06a must have been executed so that PRED_BASELINE_VW exists.

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

# Target tables (baseline only)
PRED_DRIFT_HISTOGRAMS_BASELINE = "DA_PREDICTION_DRIFT_HISTOGRAMS_BASELINE"
PRED_BASELINE_VW = "DA_PREDICTIONS_BASELINE_VW"

# Model specifications
MODEL_NAME = "UNI_BOX_REGRESSION_PARTITIONED"
ID_COLS = ["customer_id", "brand_pres_ret", "prod_key"]
AGG_COLS = ["STATS_NTILE_GROUP", "CUST_CATEGORY"]
PREDICTION_COL = "PREDICTED_UNI_BOX_WEEK"
TIME_COL = "week"

# Feature drift configuration
N_BINS = 20

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
# ## 4. Prediction drift
#
# Calculates baseline histograms of the prediction distribution per segment.
# The same quantile-based bins as for features are used, but applied
# to the prediction column. These histograms serve as a reference to
# detect drift in the model predictions in production.

# %%
def get_prediction_bin_edges(src_tbl, agg_col):
    """Return per-segment bin edges for the prediction column.

    Computes N_BINS quantile-based bin edges from PREDICTION_COL,
    separately for each segment defined by *agg_col*.

    Parameters
    ----------
    src_tbl : snowpark.DataFrame
        DataFrame pointing to the baseline prediction data.
    agg_col : str
        Column name to aggregate by (e.g. "STATS_NTILE_GROUP").

    Returns
    -------
    snowpark.DataFrame
        bins_df with columns AGGREGATED_VALUE, BIN, BIN_LOW, BIN_HIGH.
    """
    quantile_fracs = [i / N_BINS for i in range(1, N_BINS)]
    agg_exprs = [
        F.call_builtin(
            "APPROX_PERCENTILE", F.col("PREDICTION").cast("FLOAT"), F.lit(q)
        ).alias(f"P{idx}")
        for idx, q in enumerate(quantile_fracs)
    ]

    pct_wide = (
        src_tbl
        .group_by(F.col(agg_col).cast("STRING").alias("AGGREGATED_VALUE"))
        .agg(*agg_exprs)
        .cache_result()
    )

    p_cols = [F.col(f"P{idx}") for idx in range(len(quantile_fracs))]
    interior_edges = (
        pct_wide
        .unpivot("EDGE_VALUE", "P_LABEL", p_cols)
        .select(
            F.col("AGGREGATED_VALUE"),
            (F.call_builtin("REGEXP_SUBSTR", F.col("P_LABEL"), F.lit("\\d+"))
             .cast("INT") + F.lit(1)).alias("EDGE_IDX"),
            F.col("EDGE_VALUE").cast("FLOAT").alias("EDGE_VALUE"),
        )
    )

    # Add sentinel edges: -Inf at index 0, +Inf at (max EDGE_IDX + 1)
    inf_val = float("inf")
    segments = interior_edges.select("AGGREGATED_VALUE").distinct()
    max_edge = (
        interior_edges
        .group_by("AGGREGATED_VALUE")
        .agg(F.max("EDGE_IDX").alias("MAX_IDX"))
    )

    low_sentinel = segments.select(
        F.col("AGGREGATED_VALUE"),
        F.lit(0).alias("EDGE_IDX"),
        F.lit(-inf_val).cast("FLOAT").alias("EDGE_VALUE"),
    )
    high_sentinel = max_edge.select(
        F.col("AGGREGATED_VALUE"),
        (F.col("MAX_IDX") + F.lit(1)).alias("EDGE_IDX"),
        F.lit(inf_val).cast("FLOAT").alias("EDGE_VALUE"),
    )

    all_edges = low_sentinel.union_all_by_name(
        interior_edges
    ).union_all_by_name(
        high_sentinel
    )

    # Self-join consecutive edges to form (BIN, BIN_LOW, BIN_HIGH)
    low = all_edges.select(
        F.col("AGGREGATED_VALUE"),
        F.col("EDGE_IDX").alias("BIN"),
        F.col("EDGE_VALUE").alias("BIN_LOW"),
    )
    high = all_edges.select(
        F.col("AGGREGATED_VALUE"),
        (F.col("EDGE_IDX") - F.lit(1)).alias("BIN"),
        F.col("EDGE_VALUE").alias("BIN_HIGH"),
    )

    bins = (
        low.join(high, on=["AGGREGATED_VALUE", "BIN"])
        .filter(F.col("BIN") >= F.lit(1))
        .select("AGGREGATED_VALUE", "BIN", "BIN_LOW", "BIN_HIGH")
    )

    return bins

# %%
def compute_prediction_drift_histograms(versions, data_source, target_table, agg_col):
    """Compute per-segment baseline prediction histograms and write to target_table.

    Parameters
    ----------
    versions : list[str]
        Model versions for which to compute baselines.
    data_source : str | dict[str, str]
        Fully qualified table name, or mapping {version: table}.
    target_table : str
        Fully qualified table name to write results to.
    agg_col : str
        Column name to aggregate by (e.g. "STATS_NTILE_GROUP").
    """
    agg_col_lower = agg_col.lower()
    frames = []

    for version in versions:
        src = data_source[version] if isinstance(data_source, dict) else data_source
        src_tbl = (
            session.table(src)
            .filter(F.col("MODEL_NAME") == MODEL_NAME)
            .with_column(TIME_COL, F.col("ENTITY_MAP")[TIME_COL].cast("NUMBER"))
        )

        # Get bin edges
        bins = get_prediction_bin_edges(src_tbl, agg_col)

        # Assign bins to the prediction column directly (no unpivot)
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

        # Scaffold: every (TIME x bin) so empty bins get a record
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
                        F.col("BIN").cast("STRING"), F.lit("||"),
                        F.lit("jensen-shannon"),
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
    print(f"Wrote {len(versions)} version(s) for {agg_col}. {target_table} now has {n_rows} records.")

# %%
# Compute baseline prediction histograms -- iterate over each aggregation column

for agg_col in AGG_COLS:
    new_versions = get_new_baseline_versions(
        PRED_DRIFT_HISTOGRAMS_BASELINE, agg_col=agg_col,
        metric_col="jensen-shannon")

    if new_versions:
        baseline_source = {v: PRED_BASELINE_VW for v in new_versions}
        compute_prediction_drift_histograms(
            new_versions, baseline_source,
            target_table=PRED_DRIFT_HISTOGRAMS_BASELINE, agg_col=agg_col)
