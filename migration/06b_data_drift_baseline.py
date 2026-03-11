# %% [markdown]
# # 06b — Data drift baseline
#
# Part 2 of the baseline generation pipeline.
# Calculates the baseline data distribution histograms required to detect
# drift in production. Includes two metrics:
# - Population Stability Index (PSI) per segment (section 3A)
# - Jensen-Shannon distance (JSD) per numerical feature (section 3B)
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
MODELS_SCHEMA = "SC_MODELS_BMX"

session.sql(f"USE DATABASE {DATABASE}").collect()
session.sql(f"USE SCHEMA {STORAGE_SCHEMA}").collect()

# Input data sources
FEATURE_TABLE = f"{DATABASE}.{FEATURES_SCHEMA}.TRAIN_DATASET_HOLDOUT_VW"

# Target tables (baseline only)
DATA_DRIFT_HISTOGRAMS_BASELINE = "DA_DATA_DRIFT_HISTOGRAMS_BASELINE"
PRED_BASELINE_VW = "DA_PREDICTIONS_BASELINE_VW"

# Model specifications
MODEL_NAME = "UNI_BOX_REGRESSION_PARTITIONED"
ID_COLS = ["customer_id", "brand_pres_ret", "prod_key"]
AGG_COLS = ["STATS_NTILE_GROUP", "CUST_CATEGORY"]
TARGET_COL = "UNI_BOX_WEEK"
PREDICTION_COL = "PREDICTED_UNI_BOX_WEEK"
TIME_COL = "week"

NON_FEATURE_COLS = {
    *ID_COLS, TIME_COL, TARGET_COL, *AGG_COLS,
}

NON_DRIFT_COLS = {'WEEK_OF_YEAR'}

# Feature drift configuration
N_BINS = 20

# %% [markdown]
# ### 1B. Functions

# %%
def get_new_baseline_versions(baseline_table, agg_col, metric_col):
    """Find model versions in PRED_BASELINE that are missing from a baseline table.

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
# ## 3. Data drift
#
# Calculates the baseline data distribution histograms, required to detect
# drift in production. Includes two metrics: population stability index (PSI) per segment
# and Jensen-Shannon distance (JSD) per numerical feature.

# %% [markdown]
# ### 3A. Population drift
#
# Calculates customer proportions per segment (STATS_NTILE_GROUP, CUST_CATEGORY)
# in the training data. These proportions are used as a reference for the
# PSI (Population Stability Index) calculation in production.

# %%
def compute_segment_histograms(versions, data_source, target_table, agg_col):
    """Compute customer proportions per segment and write to target_table.

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
        src_tbl = session.table(src)

        data_date_expr = F.dateadd(
            "week", F.col(TIME_COL) % F.lit(100) - F.lit(1),
            F.date_from_parts(F.floor(F.col(TIME_COL) / F.lit(100)), F.lit(1), F.lit(1)),
        )

        proportions = (
            src_tbl
            .group_by(TIME_COL, agg_col)
            # TODO: check whether count_distinct(CUSTOMER_ID) is more appropriate than counting rows
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
    print(f"Wrote {len(versions)} version(s) for {agg_col}. {target_table} now has {n_rows} records.")

# %%
# Identifies the model versions that need baseline information and
# generates the baseline histograms

for agg_col in AGG_COLS:
    new_versions = get_new_baseline_versions(
        DATA_DRIFT_HISTOGRAMS_BASELINE, agg_col=agg_col,
        metric_col="population_stability_index")

    if new_versions:
        baseline_source = {v: FEATURE_TABLE for v in new_versions}
        compute_segment_histograms(
            new_versions, baseline_source,
            target_table=DATA_DRIFT_HISTOGRAMS_BASELINE, agg_col=agg_col)

# %% [markdown]
# ### 3B. Feature drift
#
# Generates base histograms per numerical feature and per segment. For each feature,
# bin edges are calculated (quantiles for high cardinality, midpoints for low cardinality)
# and observations are counted per bin. These histograms serve as a reference distribution
# to calculate the Jensen-Shannon distance in production.

# %%
def get_feature_bin_edges(src_tbl, agg_col):
    """Return per-feature per-segment bin edges and the list of monitored features.

    Identifies numeric feature columns from the data source schema
    (excluding non-feature columns), then computes bin edges
    separately for each segment. Features with fewer than N_BINS
    distinct values get one bin per distinct value (edges at midpoints
    between sorted values, bounded by -Inf / +Inf). Features with
    N_BINS or more distinct values use N_BINS quantile-based cut points.

    Parameters
    ----------
    src_tbl : snowpark.DataFrame
        DataFrame pointing to the baseline data.
    agg_col : str
        Column name to aggregate by (e.g. "STATS_NTILE_GROUP").

    Returns
    -------
    tuple[snowpark.DataFrame, list[str]]
        (bins_df, features) where bins_df has columns
        AGGREGATED_VALUE, FEATURE_NAME, BIN, BIN_LOW, BIN_HIGH;
        and features is the list of monitored feature names.
    """
    from snowflake.snowpark.types import (
        DecimalType, DoubleType, FloatType, IntegerType, LongType, ShortType,
    )
    numeric_types = (DecimalType, DoubleType, FloatType, IntegerType, LongType, ShortType)
    features = [
        field.name
        for field in src_tbl.schema.fields
        if isinstance(field.datatype, numeric_types)
        and field.name not in NON_FEATURE_COLS.union(NON_DRIFT_COLS)
    ]

    # Exclude dummy variables (features with <= 2 observed distinct values)
    distinct_counts = (
        src_tbl
        .select([F.count_distinct(F.col(f)).alias(f) for f in features])
        .collect()[0]
    )
    features = [f for f in features if distinct_counts[f] > 2]

    # Split into low cardinality (< N_BINS global distinct values)
    # and high cardinality to only unpivot what each path needs.
    low_card_features = [f for f in features if distinct_counts[f] < N_BINS]
    high_card_features = [f for f in features if distinct_counts[f] >= N_BINS]

    edge_parts = []

    # -- Low cardinality: midpoint edges between consecutive distinct values --
    if low_card_features:
        lc_cast = [F.col(f).cast("FLOAT").alias(f) for f in low_card_features]
        lc_casted = src_tbl.select(F.col(agg_col), *lc_cast)

        lc_unpivoted = (
            lc_casted.unpivot(
                "FEATURE_VALUE", "FEATURE_NAME",
                [F.col(f) for f in low_card_features],
            )
            .select(
                F.col(agg_col).cast("STRING").alias("AGGREGATED_VALUE"),
                "FEATURE_NAME", "FEATURE_VALUE",
            )
            .filter(F.col("FEATURE_VALUE").is_not_null())
        )

        low_card_vals = lc_unpivoted.select(
            "AGGREGATED_VALUE", "FEATURE_NAME", "FEATURE_VALUE"
        ).distinct()

        val_rank = Window.partition_by("AGGREGATED_VALUE", "FEATURE_NAME").order_by("FEATURE_VALUE")
        low_card_ranked = low_card_vals.with_column(
            "VAL_IDX", F.row_number().over(val_rank)
        )

        curr = low_card_ranked.select(
            F.col("AGGREGATED_VALUE"), F.col("FEATURE_NAME"),
            F.col("VAL_IDX"),
            F.col("FEATURE_VALUE").alias("CURR_VAL"),
        )
        nxt = low_card_ranked.select(
            F.col("AGGREGATED_VALUE"), F.col("FEATURE_NAME"),
            (F.col("VAL_IDX") - F.lit(1)).alias("VAL_IDX"),
            F.col("FEATURE_VALUE").alias("NEXT_VAL"),
        )
        midpoints = (
            curr.join(nxt, on=["AGGREGATED_VALUE", "FEATURE_NAME", "VAL_IDX"])
            .select(
                "AGGREGATED_VALUE", "FEATURE_NAME",
                F.col("VAL_IDX").alias("EDGE_IDX"),
                ((F.col("CURR_VAL") + F.col("NEXT_VAL")) / F.lit(2.0))
                    .cast("FLOAT").alias("EDGE_VALUE"),
            )
        )
        edge_parts.append(midpoints)

    # -- High cardinality: quantile-based edges --
    if high_card_features:
        hc_cast = [F.col(f).cast("FLOAT").alias(f) for f in high_card_features]
        hc_casted = src_tbl.select(F.col(agg_col), *hc_cast)

        hc_unpivoted = (
            hc_casted.unpivot(
                "FEATURE_VALUE", "FEATURE_NAME",
                [F.col(f) for f in high_card_features],
            )
            .select(
                F.col(agg_col).cast("STRING").alias("AGGREGATED_VALUE"),
                "FEATURE_NAME", "FEATURE_VALUE",
            )
            .filter(F.col("FEATURE_VALUE").is_not_null())
        )

        quantile_fracs = [i / N_BINS for i in range(1, N_BINS)]
        agg_exprs = [
            F.call_builtin(
                "APPROX_PERCENTILE", F.col("FEATURE_VALUE"), F.lit(q)
            ).alias(f"P{idx}")
            for idx, q in enumerate(quantile_fracs)
        ]

        pct_wide = (
            hc_unpivoted
            .group_by("AGGREGATED_VALUE", "FEATURE_NAME")
            .agg(*agg_exprs)
            .cache_result()
        )

        p_cols = [F.col(f"P{idx}") for idx in range(len(quantile_fracs))]
        quantile_edges = (
            pct_wide
            .unpivot("EDGE_VALUE", "P_LABEL", p_cols)
            .select(
                F.col("AGGREGATED_VALUE"),
                F.col("FEATURE_NAME"),
                (F.call_builtin("REGEXP_SUBSTR", F.col("P_LABEL"), F.lit("\\d+"))
                 .cast("INT") + F.lit(1)).alias("EDGE_IDX"),
                F.col("EDGE_VALUE").cast("FLOAT").alias("EDGE_VALUE"),
            )
        )
        edge_parts.append(quantile_edges)

    # -- Combine interior edges from both paths --
    interior_edges = edge_parts[0]
    for part in edge_parts[1:]:
        interior_edges = interior_edges.union_all_by_name(part)

    # Add sentinel edges: -Inf at index 0, +Inf at (max EDGE_IDX + 1)
    inf_val = float("inf")
    seg_feat = interior_edges.select("AGGREGATED_VALUE", "FEATURE_NAME").distinct()
    max_edge = (
        interior_edges
        .group_by("AGGREGATED_VALUE", "FEATURE_NAME")
        .agg(F.max("EDGE_IDX").alias("MAX_IDX"))
    )

    low_sentinel = seg_feat.select(
        F.col("AGGREGATED_VALUE"), F.col("FEATURE_NAME"),
        F.lit(0).alias("EDGE_IDX"),
        F.lit(-inf_val).cast("FLOAT").alias("EDGE_VALUE"),
    )
    high_sentinel = max_edge.select(
        F.col("AGGREGATED_VALUE"), F.col("FEATURE_NAME"),
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
        F.col("AGGREGATED_VALUE"), F.col("FEATURE_NAME"),
        F.col("EDGE_IDX").alias("BIN"),
        F.col("EDGE_VALUE").alias("BIN_LOW"),
    )
    high = all_edges.select(
        F.col("AGGREGATED_VALUE"), F.col("FEATURE_NAME"),
        (F.col("EDGE_IDX") - F.lit(1)).alias("BIN"),
        F.col("EDGE_VALUE").alias("BIN_HIGH"),
    )

    bins = (
        low.join(high, on=["AGGREGATED_VALUE", "FEATURE_NAME", "BIN"])
        .filter(F.col("BIN") >= F.lit(1))
        .select("AGGREGATED_VALUE", "FEATURE_NAME", "BIN", "BIN_LOW", "BIN_HIGH")
    )

    return bins, features

# %%
def compute_data_drift_histograms(versions, data_source, target_table, agg_col):
    """Compute per-feature per-segment baseline histograms and write to target_table.

    Uses bin edges from get_feature_bin_edges computed from the data.
    Each bin gets one record per time period, even when
    there are no values in that bin (BIN_COUNT = 0).

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
        src_tbl = session.table(src)

        # Get bin edges and the list of features to monitor
        bins, features = get_feature_bin_edges(src_tbl, agg_col)

        # Unpivot source data to (TIME, SEGMENT, FEATURE_NAME, FEATURE_VALUE)
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

        # Count values per bin (only bins with data)
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

        # Build scaffold: every (TIME x bin) combination so empty bins get a record
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

        # Build output in long format
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
    print(f"Wrote {len(versions)} version(s) for {agg_col}. {target_table} now has {n_rows} records.")

# %%
# Compute baseline feature histograms -- iterate over each aggregation column

for agg_col in AGG_COLS:
    new_versions = get_new_baseline_versions(
        DATA_DRIFT_HISTOGRAMS_BASELINE, agg_col=agg_col,
        metric_col="jensen-shannon")

    if new_versions:
        baseline_source = {v: FEATURE_TABLE for v in new_versions}
        compute_data_drift_histograms(
            new_versions, baseline_source,
            target_table=DATA_DRIFT_HISTOGRAMS_BASELINE, agg_col=agg_col)
