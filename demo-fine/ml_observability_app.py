import streamlit as st
import pandas as pd
import altair as alt
from snowflake.snowpark.context import get_active_session

st.title(":material/monitoring: ML Observability Dashboard")
st.caption("ARCA Beverage Demo - MLOps Monitoring")

session = get_active_session()

@st.cache_data(ttl=60)
def get_inference_summary():
    return session.sql("""
        SELECT
            SEGMENT,
            COUNT(*) AS PREDICTIONS,
            ROUND(AVG(PREDICTED_WEEKLY_SALES), 2) AS AVG_PREDICTED,
            ROUND(AVG(ACTUAL_WEEKLY_SALES), 2) AS AVG_ACTUAL,
            ROUND(AVG(ABS(PREDICTION_ERROR)), 2) AS MAE,
            ROUND(SQRT(AVG(POWER(PREDICTION_ERROR, 2))), 2) AS RMSE,
            ROUND(AVG(ABS(PREDICTION_ERROR / NULLIF(ACTUAL_WEEKLY_SALES, 0))) * 100, 1) AS MAPE_PCT
        FROM ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_LOGS
        GROUP BY SEGMENT
        ORDER BY SEGMENT
    """).to_pandas()

@st.cache_data(ttl=60)
def get_drift_metrics(monitor_name: str):
    try:
        return session.sql(f"""
            SELECT *
            FROM TABLE(MODEL_MONITOR_DRIFT_METRIC(
                'ARCA_BEVERAGE_DEMO.MODEL_REGISTRY.{monitor_name}',
                'JENSEN_SHANNON',
                'CUSTOMER_TOTAL_UNITS_4W',
                '1 DAY',
                DATEADD('DAY', -30, CURRENT_DATE()),
                CURRENT_DATE()
            ))
        """).to_pandas()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_performance_metrics(monitor_name: str, metric: str = 'RMSE'):
    try:
        return session.sql(f"""
            SELECT *
            FROM TABLE(MODEL_MONITOR_PERFORMANCE_METRIC(
                'ARCA_BEVERAGE_DEMO.MODEL_REGISTRY.{monitor_name}',
                '{metric}',
                '1 DAY',
                DATEADD('DAY', -30, CURRENT_DATE()),
                CURRENT_DATE()
            ))
        """).to_pandas()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_distribution_comparison():
    return session.sql("""
        WITH baseline AS (
            SELECT 'SEGMENT_1' AS SEGMENT, 'BASELINE' AS SOURCE,
                AVG(CUSTOMER_TOTAL_UNITS_4W) AS AVG_UNITS,
                STDDEV(CUSTOMER_TOTAL_UNITS_4W) AS STDDEV_UNITS
            FROM ARCA_BEVERAGE_DEMO.ML_DATA.BASELINE_SEGMENT_1
            UNION ALL
            SELECT 'SEGMENT_3', 'BASELINE', AVG(CUSTOMER_TOTAL_UNITS_4W), STDDEV(CUSTOMER_TOTAL_UNITS_4W)
            FROM ARCA_BEVERAGE_DEMO.ML_DATA.BASELINE_SEGMENT_3
            UNION ALL
            SELECT 'SEGMENT_5', 'BASELINE', AVG(CUSTOMER_TOTAL_UNITS_4W), STDDEV(CUSTOMER_TOTAL_UNITS_4W)
            FROM ARCA_BEVERAGE_DEMO.ML_DATA.BASELINE_SEGMENT_5
        ),
        inference AS (
            SELECT 'SEGMENT_1' AS SEGMENT, 'INFERENCE' AS SOURCE,
                AVG(CUSTOMER_TOTAL_UNITS_4W) AS AVG_UNITS,
                STDDEV(CUSTOMER_TOTAL_UNITS_4W) AS STDDEV_UNITS
            FROM ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_SEGMENT_1
            UNION ALL
            SELECT 'SEGMENT_3', 'INFERENCE', AVG(CUSTOMER_TOTAL_UNITS_4W), STDDEV(CUSTOMER_TOTAL_UNITS_4W)
            FROM ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_SEGMENT_3
            UNION ALL
            SELECT 'SEGMENT_5', 'INFERENCE', AVG(CUSTOMER_TOTAL_UNITS_4W), STDDEV(CUSTOMER_TOTAL_UNITS_4W)
            FROM ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_SEGMENT_5
        )
        SELECT * FROM baseline
        UNION ALL
        SELECT * FROM inference
        ORDER BY SEGMENT, SOURCE
    """).to_pandas()

@st.cache_data(ttl=60)
def get_recent_predictions():
    return session.sql("""
        SELECT 
            CUSTOMER_ID,
            SEGMENT,
            WEEK_START_DATE,
            ROUND(PREDICTED_WEEKLY_SALES, 2) AS PREDICTED,
            ROUND(ACTUAL_WEEKLY_SALES, 2) AS ACTUAL,
            ROUND(ABSOLUTE_ERROR, 2) AS ERROR,
            INFERENCE_TIMESTAMP
        FROM ARCA_BEVERAGE_DEMO.ML_DATA.INFERENCE_LOGS
        ORDER BY INFERENCE_TIMESTAMP DESC
        LIMIT 100
    """).to_pandas()

monitors = ['SEGMENT_1', 'SEGMENT_3', 'SEGMENT_5']

tab1, tab2, tab3, tab4 = st.tabs([
    ":material/dashboard: Overview", 
    ":material/trending_up: Drift Analysis",
    ":material/speed: Performance",
    ":material/table_view: Predictions"
])

with tab1:
    st.subheader("Model Performance Summary")
    
    summary_df = get_inference_summary()
    
    if not summary_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        total_predictions = summary_df['PREDICTIONS'].sum()
        avg_mae = summary_df['MAE'].mean()
        avg_rmse = summary_df['RMSE'].mean()
        avg_mape = summary_df['MAPE_PCT'].mean()
        
        col1.metric("Total Predictions", f"{total_predictions:,}")
        col2.metric("Avg MAE", f"{avg_mae:.3f}")
        col3.metric("Avg RMSE", f"{avg_rmse:.3f}")
        col4.metric("Avg MAPE", f"{avg_mape:.1f}%")
        
        st.divider()
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.write("**Performance by Segment**")
            
            chart_data = summary_df[['SEGMENT', 'MAE', 'RMSE']].melt(
                id_vars=['SEGMENT'], 
                var_name='Metric', 
                value_name='Value'
            )
            
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('SEGMENT:N', title='Segment'),
                y=alt.Y('Value:Q', title='Error Value'),
                color=alt.Color('Metric:N', scale=alt.Scale(scheme='blues')),
                xOffset='Metric:N'
            ).properties(height=300)
            
            st.altair_chart(chart, use_container_width=True)
        
        with col_right:
            st.write("**Predictions by Segment**")
            
            pie_chart = alt.Chart(summary_df).mark_arc(innerRadius=50).encode(
                theta=alt.Theta('PREDICTIONS:Q'),
                color=alt.Color('SEGMENT:N', scale=alt.Scale(scheme='category10')),
                tooltip=['SEGMENT', 'PREDICTIONS']
            ).properties(height=300)
            
            st.altair_chart(pie_chart, use_container_width=True)
        
        st.write("**Detailed Metrics**")
        
        st.dataframe(
            summary_df,
            column_config={
                "SEGMENT": st.column_config.TextColumn("Segment"),
                "PREDICTIONS": st.column_config.NumberColumn("Predictions", format="%d"),
                "AVG_PREDICTED": st.column_config.NumberColumn("Avg Predicted", format="%.2f"),
                "AVG_ACTUAL": st.column_config.NumberColumn("Avg Actual", format="%.2f"),
                "MAE": st.column_config.NumberColumn("MAE", format="%.3f"),
                "RMSE": st.column_config.NumberColumn("RMSE", format="%.3f"),
                "MAPE_PCT": st.column_config.ProgressColumn("MAPE %", min_value=0, max_value=100, format="%.1f%%")
            },
            hide_index=True,
            use_container_width=True
        )

with tab2:
    st.subheader("Data Drift Analysis")
    st.caption("Jensen-Shannon Divergence measures distribution shift between training and inference data")
    
    col1, col2, col3 = st.columns(3)
    
    drift_warning = 0.2
    drift_critical = 0.4
    
    drift_results = []
    
    for i, segment in enumerate(monitors):
        monitor_name = f'WEEKLY_SALES_{segment}_MONITOR'
        drift_df = get_drift_metrics(monitor_name)
        
        if not drift_df.empty:
            drift_value = drift_df['METRIC_VALUE'].iloc[0]
            drift_results.append({'SEGMENT': segment, 'DRIFT': drift_value})
            
            with [col1, col2, col3][i]:
                st.metric(
                    label=f"{segment}",
                    value=f"{drift_value:.3f}"
                )
                
                if drift_value < drift_warning:
                    st.success("OK", icon=":material/check_circle:")
                elif drift_value < drift_critical:
                    st.warning("WARNING", icon=":material/warning:")
                else:
                    st.error("CRITICAL", icon=":material/error:")
    
    st.divider()
    
    st.write("**Feature Distribution: Baseline vs Inference**")
    
    dist_df = get_distribution_comparison()
    
    if not dist_df.empty:
        chart = alt.Chart(dist_df).mark_bar().encode(
            x=alt.X('SEGMENT:N', title='Segment'),
            y=alt.Y('AVG_UNITS:Q', title='Avg Units (4 weeks)'),
            color=alt.Color('SOURCE:N', scale=alt.Scale(domain=['BASELINE', 'INFERENCE'], range=['#1f77b4', '#ff7f0e'])),
            xOffset='SOURCE:N'
        ).properties(height=350)
        
        st.altair_chart(chart, use_container_width=True)
    
    st.divider()
    
    st.write("**Drift Thresholds**")
    threshold_col1, threshold_col2, threshold_col3 = st.columns(3)
    threshold_col1.info(":material/check_circle: **OK**: JS < 0.2")
    threshold_col2.warning(":material/warning: **Warning**: JS 0.2 - 0.4")
    threshold_col3.error(":material/error: **Critical**: JS > 0.4")

with tab3:
    st.subheader("Model Performance Metrics")
    
    metric_options = ['RMSE', 'MAE', 'MAPE']
    selected_metric = st.segmented_control("Select Metric", metric_options, default='RMSE')
    
    perf_results = []
    
    for segment in monitors:
        monitor_name = f'WEEKLY_SALES_{segment}_MONITOR'
        perf_df = get_performance_metrics(monitor_name, selected_metric)
        
        if not perf_df.empty:
            perf_results.append({
                'SEGMENT': segment,
                'METRIC': selected_metric,
                'VALUE': perf_df['METRIC_VALUE'].iloc[0],
                'COUNT': perf_df['COUNT_USED'].iloc[0]
            })
    
    if perf_results:
        perf_results_df = pd.DataFrame(perf_results)
        
        col1, col2, col3 = st.columns(3)
        
        for i, row in perf_results_df.iterrows():
            with [col1, col2, col3][i]:
                st.metric(
                    label=f"{row['SEGMENT']}",
                    value=f"{row['VALUE']:.4f}",
                    delta=f"{int(row['COUNT'])} predictions"
                )
        
        st.divider()
        
        chart = alt.Chart(perf_results_df).mark_bar(color='#1f77b4').encode(
            x=alt.X('SEGMENT:N', title='Segment'),
            y=alt.Y('VALUE:Q', title=selected_metric),
            tooltip=['SEGMENT', 'VALUE', 'COUNT']
        ).properties(height=300)
        
        st.altair_chart(chart, use_container_width=True)
    
    st.divider()
    
    st.write("**Performance Thresholds**")
    
    thresholds = {
        'RMSE': {'warning': 0.5, 'critical': 1.0},
        'MAE': {'warning': 0.5, 'critical': 1.0},
        'MAPE': {'warning': 15.0, 'critical': 25.0}
    }
    
    thresh = thresholds[selected_metric]
    t_col1, t_col2, t_col3 = st.columns(3)
    t_col1.success(f":material/check_circle: **Good**: < {thresh['warning']}")
    t_col2.warning(f":material/warning: **Warning**: {thresh['warning']} - {thresh['critical']}")
    t_col3.error(f":material/error: **Critical**: > {thresh['critical']}")

with tab4:
    st.subheader("Recent Predictions")
    
    predictions_df = get_recent_predictions()
    
    if not predictions_df.empty:
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            segments = ['All'] + list(predictions_df['SEGMENT'].unique())
            selected_segment = st.selectbox("Filter by Segment", segments)
        
        with filter_col2:
            max_error = st.slider("Max Error Filter", 0.0, float(predictions_df['ERROR'].max()), float(predictions_df['ERROR'].max()))
        
        filtered_df = predictions_df.copy()
        if selected_segment != 'All':
            filtered_df = filtered_df[filtered_df['SEGMENT'] == selected_segment]
        filtered_df = filtered_df[filtered_df['ERROR'] <= max_error]
        
        st.dataframe(
            filtered_df,
            column_config={
                "CUSTOMER_ID": st.column_config.NumberColumn("Customer ID"),
                "SEGMENT": st.column_config.TextColumn("Segment"),
                "WEEK_START_DATE": st.column_config.DateColumn("Week"),
                "PREDICTED": st.column_config.NumberColumn("Predicted", format="%.2f"),
                "ACTUAL": st.column_config.NumberColumn("Actual", format="%.2f"),
                "ERROR": st.column_config.NumberColumn("Abs Error", format="%.2f"),
                "INFERENCE_TIMESTAMP": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm")
            },
            hide_index=True,
            use_container_width=True
        )
        
        st.caption(f"Showing {len(filtered_df)} of {len(predictions_df)} predictions")

st.divider()

with st.expander(":material/info: About this Dashboard"):
    st.markdown("""
    ### ML Observability Dashboard
    
    This dashboard monitors the ARCA Beverage sales forecasting models:
    
    **Metrics Tracked:**
    - **Drift (Jensen-Shannon)**: Measures distribution shift between training and inference data
    - **RMSE/MAE**: Root Mean Square Error and Mean Absolute Error
    - **MAPE**: Mean Absolute Percentage Error
    
    **Alert Thresholds:**
    | Metric | Warning | Critical |
    |--------|---------|----------|
    | Drift (JS) | > 0.2 | > 0.4 |
    | RMSE/MAE | > 0.5 | > 1.0 |
    | MAPE | > 15% | > 25% |
    
    **Models Monitored:**
    - SEGMENT_1: High Frequency - High Volume
    - SEGMENT_3: Medium Frequency - High Volume  
    - SEGMENT_5: Low Frequency - Low Volume
    """)
