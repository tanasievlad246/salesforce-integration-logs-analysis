"""Streamlit dashboard for Salesforce Integration Log Analytics."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

DB_PATH = Path(__file__).parent.parent / "logs_analysis.db"


def calculate_apdex(df: pd.DataFrame, threshold_ms: int = 500) -> float:
    """Calculate Apdex score for the given dataframe.

    Apdex (Application Performance Index) formula:
    - Satisfied: duration < T
    - Tolerating: T <= duration < 4T
    - Frustrated: duration >= 4T
    - Score = (Satisfied + 0.5 * Tolerating) / Total
    """
    if len(df) == 0:
        return 0.0
    satisfied = (df["duration_ms"] < threshold_ms).sum()
    tolerating = (
        (df["duration_ms"] >= threshold_ms) & (df["duration_ms"] < 4 * threshold_ms)
    ).sum()
    total = len(df)
    return (satisfied + 0.5 * tolerating) / total


def get_apdex_color_and_label(score: float) -> tuple[str, str]:
    """Get color and label for Apdex score."""
    if score >= 0.94:
        return "#28a745", "Excellent"
    elif score >= 0.85:
        return "#17a2b8", "Good"
    elif score >= 0.70:
        return "#ffc107", "Fair"
    elif score >= 0.50:
        return "#fd7e14", "Poor"
    else:
        return "#dc3545", "Unacceptable"


@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    """Load all action metrics from the database."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT
            timestamp, name, duration_ms, handler, client, operation,
            request_id, operation_id, flow, level, app_version, source_file
        FROM action_metrics
        WHERE duration_ms IS NOT NULL
        """,
        conn,
        parse_dates=["timestamp"],
    )
    conn.close()
    return df


@st.cache_data
def get_filter_options(df: pd.DataFrame) -> dict:
    """Get unique values for filter dropdowns."""
    return {
        "clients": sorted(df["client"].dropna().unique().tolist()),
        "handlers": sorted(df["handler"].dropna().unique().tolist()),
        "operations": sorted(df["operation"].dropna().unique().tolist()),
    }


# =============================================================================
# NEW STATISTICS FUNCTIONS
# =============================================================================


def render_ops_per_request(requests_df: pd.DataFrame, operations_df: pd.DataFrame):
    """Analyze operations triggered per HTTP request."""
    st.subheader("Operations per Request")

    if len(requests_df) == 0 or len(operations_df) == 0:
        st.warning("No data available for ops per request analysis.")
        return

    # Count operations per request_id
    ops_per_req = operations_df.groupby("request_id").size().reset_index(name="ops_count")

    if len(ops_per_req) == 0:
        st.warning("No request_id correlations found.")
        return

    # Calculate metrics
    avg_ops = ops_per_req["ops_count"].mean()
    min_ops = ops_per_req["ops_count"].min()
    max_ops = ops_per_req["ops_count"].max()
    median_ops = ops_per_req["ops_count"].median()

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Ops/Request", f"{avg_ops:.1f}")
    with col2:
        st.metric("Median Ops/Request", f"{median_ops:.0f}")
    with col3:
        st.metric("Min Ops", f"{min_ops:.0f}")
    with col4:
        st.metric("Max Ops", f"{max_ops:.0f}")

    # Two columns: histogram and top handlers
    col1, col2 = st.columns(2)

    with col1:
        # Histogram of ops distribution
        fig = px.histogram(
            ops_per_req,
            x="ops_count",
            nbins=min(30, ops_per_req["ops_count"].nunique()),
            labels={"ops_count": "Operations per Request", "count": "Frequency"},
            color_discrete_sequence=["steelblue"],
        )
        fig.update_layout(
            title="Operations Distribution per Request",
            height=300,
            margin=dict(l=40, r=40, t=50, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Top handlers by ops count
        # Join operations with requests to get handler info
        ops_with_handler = operations_df.merge(
            requests_df[["request_id", "handler"]].drop_duplicates(),
            on="request_id",
            how="left",
            suffixes=("", "_req"),
        )
        handler_ops = ops_with_handler.groupby("handler").size().reset_index(name="total_ops")
        handler_requests = requests_df.groupby("handler").size().reset_index(name="request_count")
        handler_stats = handler_ops.merge(handler_requests, on="handler", how="left")
        handler_stats["avg_ops_per_request"] = handler_stats["total_ops"] / handler_stats["request_count"]
        handler_stats = handler_stats.sort_values("avg_ops_per_request", ascending=False).head(10)

        fig = px.bar(
            handler_stats,
            x="avg_ops_per_request",
            y="handler",
            orientation="h",
            labels={"avg_ops_per_request": "Avg Ops/Request", "handler": "Handler"},
            color="avg_ops_per_request",
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            title="Top Handlers by Ops/Request",
            height=300,
            yaxis=dict(categoryorder="total ascending"),
            margin=dict(l=40, r=40, t=50, b=40),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def get_sla_color(pct: float) -> str:
    """Get color based on SLA compliance percentage."""
    if pct >= 95:
        return "#28a745"  # green
    elif pct >= 80:
        return "#ffc107"  # yellow
    else:
        return "#dc3545"  # red


def render_sla_compliance(
    df: pd.DataFrame, threshold_ms: int = 1000, is_requests: bool = True
):
    """Render SLA compliance metrics by client."""
    st.subheader("Client SLA Compliance")

    # Explanation of SLA
    with st.expander("What is SLA Compliance?", expanded=False):
        st.markdown("""
**SLA (Service Level Agreement)** is a commitment to meet specific performance targets.

In this context, SLA Compliance measures the percentage of operations that complete
within the configured threshold time. For example, with a 1000ms threshold:
- **95%+ compliance** (green): Excellent - nearly all operations meet the target
- **80-95% compliance** (yellow): Acceptable - most operations meet the target
- **Below 80%** (red): Needs attention - too many slow operations

**How to use this data:**
- Identify clients with low SLA compliance for optimization
- Adjust the threshold in the sidebar to match your actual SLA requirements
- Track compliance over time to measure improvement efforts
        """)

    if len(df) == 0:
        st.warning("No data available for SLA analysis.")
        return

    entity_name = "Requests" if is_requests else "Operations"

    # Calculate overall SLA compliance first (always possible if df is not empty)
    total_all = len(df)
    under_threshold_all = (df["duration_ms"] < threshold_ms).sum()
    overall_sla = (under_threshold_all / total_all * 100) if total_all > 0 else 0

    # Calculate compliance per client
    clients = df["client"].dropna().unique()

    if len(clients) == 0:
        # No client data available, just show the gauge
        sla_color = get_sla_color(overall_sla)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall_sla,
            number={"suffix": "%"},
            title={"text": f"Overall SLA Compliance<br><span style='font-size:12px'>{entity_name} < {threshold_ms}ms</span>"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": sla_color},
                "steps": [
                    {"range": [0, 80], "color": "#ffebee"},
                    {"range": [80, 95], "color": "#fff8e1"},
                    {"range": [95, 100], "color": "#e8f5e9"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.75,
                    "value": 95,
                },
            },
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.info("No client breakdown available (client field is empty).")
        return

    compliance_data = []
    for client in clients:
        client_df = df[df["client"] == client]
        total = len(client_df)
        under_500ms = (client_df["duration_ms"] < 500).sum()
        under_1s = (client_df["duration_ms"] < 1000).sum()
        under_2s = (client_df["duration_ms"] < 2000).sum()
        under_threshold = (client_df["duration_ms"] < threshold_ms).sum()
        sla_pct = (under_threshold / total * 100) if total > 0 else 0

        compliance_data.append({
            "Client": client,
            "Total": total,
            "Under 500ms": under_500ms,
            "Under 1s": under_1s,
            "Under 2s": under_2s,
            "SLA %": round(sla_pct, 1),
        })

    compliance_df = pd.DataFrame(compliance_data)
    compliance_df = compliance_df.sort_values("SLA %", ascending=True)

    # Display gauge and table side by side
    col1, col2 = st.columns([1, 2])

    with col1:
        # Gauge chart for overall SLA
        sla_color = get_sla_color(overall_sla)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall_sla,
            number={"suffix": "%"},
            title={"text": f"Overall SLA Compliance<br><span style='font-size:12px'>{entity_name} < {threshold_ms}ms</span>"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": sla_color},
                "steps": [
                    {"range": [0, 80], "color": "#ffebee"},
                    {"range": [80, 95], "color": "#fff8e1"},
                    {"range": [95, 100], "color": "#e8f5e9"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.75,
                    "value": 95,
                },
            },
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # SLA table with color coding
        def color_sla(val):
            if isinstance(val, (int, float)):
                color = get_sla_color(val)
                return f"color: {color}; font-weight: bold"
            return ""

        styled_df = compliance_df.style.map(
            color_sla, subset=["SLA %"]
        )

        st.markdown(f"**SLA Threshold: {threshold_ms}ms**")
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=min(300, 35 + 35 * len(compliance_df)),
            hide_index=True,
            column_config={
                "Client": st.column_config.TextColumn("Client"),
                "Total": st.column_config.NumberColumn("Total", format="%d"),
                "Under 500ms": st.column_config.NumberColumn("<500ms", format="%d"),
                "Under 1s": st.column_config.NumberColumn("<1s", format="%d"),
                "Under 2s": st.column_config.NumberColumn("<2s", format="%d"),
                "SLA %": st.column_config.NumberColumn("SLA %", format="%.1f%%"),
            },
        )


def calculate_period_comparison(
    df: pd.DataFrame, current_start: pd.Timestamp, current_end: pd.Timestamp
) -> dict | None:
    """Calculate metrics for current vs previous period."""
    if len(df) == 0:
        return None

    # Calculate period duration
    period_duration = current_end - current_start

    # Calculate previous period boundaries
    prev_end = current_start
    prev_start = prev_end - period_duration

    # Filter data for each period
    current_df = df[(df["timestamp"] >= current_start) & (df["timestamp"] <= current_end)]
    previous_df = df[(df["timestamp"] >= prev_start) & (df["timestamp"] < prev_end)]

    if len(current_df) == 0 or len(previous_df) == 0:
        return None

    # Calculate metrics for both periods
    def calc_metrics(period_df):
        return {
            "count": len(period_df),
            "avg_duration": period_df["duration_ms"].mean(),
            "p95": period_df["duration_ms"].quantile(0.95),
            "slow_pct": (period_df["duration_ms"] > 1000).mean() * 100,
        }

    current_metrics = calc_metrics(current_df)
    previous_metrics = calc_metrics(previous_df)

    # Calculate deltas
    def calc_delta(current, previous):
        if previous == 0:
            return 0
        return ((current - previous) / previous) * 100

    return {
        "current": current_metrics,
        "previous": previous_metrics,
        "deltas": {
            "count": calc_delta(current_metrics["count"], previous_metrics["count"]),
            "avg_duration": calc_delta(current_metrics["avg_duration"], previous_metrics["avg_duration"]),
            "p95": calc_delta(current_metrics["p95"], previous_metrics["p95"]),
            "slow_pct": calc_delta(current_metrics["slow_pct"], previous_metrics["slow_pct"]),
        },
        "prev_start": prev_start,
        "prev_end": prev_end,
    }


def render_period_comparison(comparison: dict | None, is_requests: bool = True):
    """Display period comparison metrics."""
    if comparison is None:
        st.info("Not enough data for period comparison (requires data in previous equivalent period).")
        return

    st.markdown("#### Period Comparison")
    st.caption(
        f"Comparing current selection against previous period "
        f"({comparison['prev_start'].strftime('%H:%M')} - {comparison['prev_end'].strftime('%H:%M')})"
    )

    entity_name = "Requests" if is_requests else "Operations"

    def get_delta_color(delta: float, invert: bool = False) -> str:
        """Get color for delta (green=improvement, red=regression)."""
        if invert:
            # For metrics where lower is better (duration, slow_pct)
            return "#28a745" if delta < 0 else "#dc3545" if delta > 0 else "#6c757d"
        else:
            # For metrics where higher is better (count)
            return "#28a745" if delta > 0 else "#dc3545" if delta < 0 else "#6c757d"

    def format_delta(delta: float) -> str:
        sign = "+" if delta > 0 else ""
        return f"{sign}{delta:.1f}%"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta = comparison["deltas"]["count"]
        color = get_delta_color(delta, invert=False)
        st.metric(
            f"{entity_name} Count",
            f"{comparison['current']['count']:,}",
            delta=f"{format_delta(delta)} vs prev",
            delta_color="normal" if delta >= 0 else "inverse",
        )

    with col2:
        delta = comparison["deltas"]["avg_duration"]
        color = get_delta_color(delta, invert=True)
        st.metric(
            "Avg Duration",
            f"{comparison['current']['avg_duration']:.0f} ms",
            delta=f"{format_delta(delta)} vs prev",
            delta_color="inverse" if delta >= 0 else "normal",
        )

    with col3:
        delta = comparison["deltas"]["p95"]
        color = get_delta_color(delta, invert=True)
        st.metric(
            "P95",
            f"{comparison['current']['p95']:.0f} ms",
            delta=f"{format_delta(delta)} vs prev",
            delta_color="inverse" if delta >= 0 else "normal",
        )

    with col4:
        delta = comparison["deltas"]["slow_pct"]
        color = get_delta_color(delta, invert=True)
        st.metric(
            "Slow %",
            f"{comparison['current']['slow_pct']:.1f}%",
            delta=f"{format_delta(delta)} vs prev",
            delta_color="inverse" if delta >= 0 else "normal",
        )


def render_slow_request_breakdown(
    requests_df: pd.DataFrame, operations_df: pd.DataFrame, threshold_ms: int = 1000
):
    """Analyze what causes slow requests."""
    st.subheader("Slow Request Breakdown")

    if len(requests_df) == 0:
        st.warning("No request data available.")
        return

    # Filter slow requests
    slow_requests = requests_df[requests_df["duration_ms"] > threshold_ms]

    if len(slow_requests) == 0:
        st.success(f"No requests exceeding {threshold_ms}ms threshold.")
        return

    st.markdown(f"**{len(slow_requests):,}** requests exceed {threshold_ms}ms threshold ({len(slow_requests)/len(requests_df)*100:.1f}% of total)")

    # Find operations associated with slow requests
    slow_request_ids = slow_requests["request_id"].dropna().unique()
    slow_ops = operations_df[operations_df["request_id"].isin(slow_request_ids)]

    if len(slow_ops) == 0:
        st.info("No operations data correlated with slow requests.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Stacked bar: Time breakdown by operation type for slow requests
        # Extract operation category
        def get_op_category(name: str) -> str:
            if pd.isna(name):
                return "Other"
            parts = name.split(".")
            if len(parts) >= 1:
                cat = parts[0]
                return cat.title() if cat else "Other"
            return "Other"

        slow_ops_with_cat = slow_ops.copy()
        slow_ops_with_cat["category"] = slow_ops_with_cat["name"].apply(get_op_category)

        # Aggregate by category
        cat_time = slow_ops_with_cat.groupby("category")["duration_ms"].sum().reset_index()
        cat_time = cat_time.sort_values("duration_ms", ascending=False).head(10)

        fig = px.bar(
            cat_time,
            x="duration_ms",
            y="category",
            orientation="h",
            labels={"duration_ms": "Total Time (ms)", "category": "Operation Category"},
            color="duration_ms",
            color_continuous_scale="Reds",
        )
        fig.update_layout(
            title="Time by Operation Category (Slow Requests)",
            height=350,
            yaxis=dict(categoryorder="total ascending"),
            margin=dict(l=40, r=40, t=50, b=40),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Table: Top contributing operations to slow requests
        op_contrib = slow_ops.groupby("name").agg(
            total_time=("duration_ms", "sum"),
            count=("duration_ms", "count"),
            avg_time=("duration_ms", "mean"),
        ).reset_index()
        op_contrib = op_contrib.sort_values("total_time", ascending=False).head(10)
        op_contrib["avg_time"] = op_contrib["avg_time"].round(0).astype(int)

        st.markdown("**Top Contributing Operations**")
        st.dataframe(
            op_contrib,
            use_container_width=True,
            height=300,
            hide_index=True,
            column_config={
                "name": st.column_config.TextColumn("Operation"),
                "total_time": st.column_config.NumberColumn("Total Time (ms)", format="%d"),
                "count": st.column_config.NumberColumn("Count", format="%d"),
                "avg_time": st.column_config.NumberColumn("Avg (ms)", format="%d"),
            },
        )

    # Heatmap: Handler x Client combinations with high slow request rates
    st.markdown("**Slow Request Heatmap by Handler × Client**")

    # Calculate slow request rate per handler-client combination
    requests_with_slow = requests_df.copy()
    requests_with_slow["is_slow"] = requests_with_slow["duration_ms"] > threshold_ms
    heatmap_data = requests_with_slow.groupby(["handler", "client"]).agg(
        total=("duration_ms", "count"),
        slow=("is_slow", "sum"),
    ).reset_index()
    heatmap_data["slow_rate"] = (heatmap_data["slow"] / heatmap_data["total"] * 100)

    if len(heatmap_data) > 0:
        # Pivot for heatmap
        heatmap_pivot = heatmap_data.pivot(
            index="handler", columns="client", values="slow_rate"
        ).fillna(0)

        fig = px.imshow(
            heatmap_pivot,
            labels=dict(x="Client", y="Handler", color="Slow %"),
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
        )
        fig.update_layout(
            height=max(300, 30 * len(heatmap_pivot)),
            margin=dict(l=40, r=40, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_throughput_analysis(df: pd.DataFrame, time_range_minutes: float, is_requests: bool = True):
    """Advanced throughput metrics with variance."""
    st.subheader("Throughput Analysis")

    if len(df) == 0:
        st.warning("No data available for throughput analysis.")
        return

    entity_name = "Requests" if is_requests else "Operations"

    # Resample by minute
    df_ts = df.set_index("timestamp").resample("1min").agg(
        count=("duration_ms", "count"),
        avg_duration=("duration_ms", "mean"),
    ).reset_index()

    if len(df_ts) == 0:
        st.warning("Not enough data points for throughput analysis.")
        return

    # Calculate throughput metrics
    throughput_series = df_ts["count"]
    peak_throughput = throughput_series.max()
    avg_throughput = throughput_series.mean()
    throughput_std = throughput_series.std()
    throughput_variance_pct = (throughput_std / avg_throughput * 100) if avg_throughput > 0 else 0

    # Calculate trend (simple linear regression slope)
    if len(df_ts) > 1:
        x = np.arange(len(df_ts))
        y = throughput_series.values
        slope, _ = np.polyfit(x, y, 1)
        trend_direction = "↑ Increasing" if slope > 0.1 else "↓ Decreasing" if slope < -0.1 else "→ Stable"
        trend_color = "#28a745" if slope > 0.1 else "#dc3545" if slope < -0.1 else "#6c757d"
    else:
        trend_direction = "→ Stable"
        trend_color = "#6c757d"
        slope = 0

    # Display KPI metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(f"Avg {entity_name}/min", f"{avg_throughput:.1f}")

    with col2:
        st.metric("Peak Throughput", f"{peak_throughput:.0f}/min")

    with col3:
        st.metric("Throughput StdDev", f"±{throughput_std:.1f}")
        st.caption(f"Variance: {throughput_variance_pct:.1f}%")

    with col4:
        st.metric("Trend", trend_direction)
        st.markdown(
            f"<p style='margin-top:-10px;color:{trend_color};font-size:12px;'>Slope: {slope:.2f}/min</p>",
            unsafe_allow_html=True,
        )

    # Time series with variance bands
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add variance band (mean ± std)
    upper_band = avg_throughput + throughput_std
    lower_band = max(0, avg_throughput - throughput_std)

    fig.add_trace(
        go.Scatter(
            x=df_ts["timestamp"],
            y=[upper_band] * len(df_ts),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df_ts["timestamp"],
            y=[lower_band] * len(df_ts),
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(70, 130, 180, 0.2)",
            name="±1 Std Dev",
            hoverinfo="skip",
        ),
        secondary_y=False,
    )

    # Add throughput line
    fig.add_trace(
        go.Scatter(
            x=df_ts["timestamp"],
            y=df_ts["count"],
            mode="lines+markers",
            name=f"{entity_name}/min",
            line=dict(color="steelblue", width=2),
            marker=dict(size=4),
        ),
        secondary_y=False,
    )

    # Add average line
    fig.add_trace(
        go.Scatter(
            x=[df_ts["timestamp"].min(), df_ts["timestamp"].max()],
            y=[avg_throughput, avg_throughput],
            mode="lines",
            name="Average",
            line=dict(color="green", width=1, dash="dash"),
        ),
        secondary_y=False,
    )

    # Add trend line
    if len(df_ts) > 1:
        trend_y = slope * x + (avg_throughput - slope * len(x) / 2)
        fig.add_trace(
            go.Scatter(
                x=df_ts["timestamp"],
                y=trend_y,
                mode="lines",
                name="Trend",
                line=dict(color="orange", width=1, dash="dot"),
            ),
            secondary_y=False,
        )

    fig.update_layout(
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text=f"{entity_name}/min", secondary_y=False)

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# END OF NEW STATISTICS FUNCTIONS
# =============================================================================


def render_data_overview(df: pd.DataFrame):
    """Render data overview panel in sidebar."""
    st.sidebar.header("Data Overview")

    # Time range
    min_time = df["timestamp"].min()
    max_time = df["timestamp"].max()
    duration_hours = (max_time - min_time).total_seconds() / 3600

    st.sidebar.markdown(f"**Time Range:**")
    st.sidebar.markdown(f"{min_time.strftime('%Y-%m-%d %H:%M')} to")
    st.sidebar.markdown(f"{max_time.strftime('%Y-%m-%d %H:%M')}")
    st.sidebar.markdown(f"*({duration_hours:.1f} hours of activity)*")

    # App version
    app_versions = df["app_version"].dropna().unique()
    if len(app_versions) > 0:
        st.sidebar.markdown(f"**App Version:** {', '.join(app_versions)}")

    # Source files
    source_files = df["source_file"].dropna().unique()
    if len(source_files) > 0:
        with st.sidebar.expander("Source Files"):
            for f in source_files:
                st.markdown(f"- {f}")

    # Quick summary
    requests_count = len(df[df["name"] == "request"])
    operations_count = len(df[df["name"] != "request"])
    st.sidebar.info(
        f"Analyzing **{duration_hours:.1f} hours** of Salesforce integration activity: "
        f"**{requests_count:,}** HTTP requests and **{operations_count:,}** operations."
    )


def render_glossary():
    """Render glossary/help section in sidebar."""
    with st.sidebar.expander("Glossary & Help"):
        st.markdown("""
**Latency Percentiles:**
- **P50 (Median):** 50% of requests are faster than this value
- **P95:** 95% of requests are faster (only 5% are slower)
- **P99:** 99% of requests are faster (only 1% are slower)

**Apdex Score:**
Application Performance Index (0-1 scale):
- **0.94-1.00:** Excellent (users are satisfied)
- **0.85-0.93:** Good (most users satisfied)
- **0.70-0.84:** Fair (some users frustrated)
- **0.50-0.69:** Poor (many users frustrated)
- **< 0.50:** Unacceptable (most users frustrated)

**Other Metrics:**
- **Throughput:** Requests or operations per minute
- **Slow (>1s):** Requests taking more than 1 second are considered slow

**How to Read the Dashboard:**
- Use filters to narrow down by time, client, or handler
- Look for handlers/operations with high P95/P99 values
- High "Cumulative Time" operations are optimization targets
        """)


def render_handler_reference(df: pd.DataFrame):
    """Render handler/route reference with counts."""
    with st.expander("Route Reference (All Handlers)"):
        handler_counts = df["handler"].value_counts().reset_index()
        handler_counts.columns = ["Handler", "Request Count"]

        # Add description based on handler name
        def get_handler_description(handler: str) -> str:
            descriptions = {
                "graphql-handler:__graphql": "GraphQL API endpoint for frontend queries",
                "public-handler:authorize": "OAuth authorization flow",
                "public-handler:accessTokenExchange": "Exchange auth code for access token",
                "public-handler:receiveAuthorizationCode": "Receive OAuth callback",
                "public-handler:logoutSalesforce": "Logout from Salesforce session",
                "public-handler:list": "List giftcards",
                "public-handler:get": "Get single resource endpoint",
                "public-handler:transactions": "Loyalty transactions endpoint",
                "public-handler:settlement": "Payment settlement processing",
                "public-handler:cancellation": "Order cancellation handling",
                "public-handler:simulateBenefitsRoute": "Simulate loyalty benefits",
                "public-handler:simulateCheckoutRoute": "Simulate checkout with benefits",
                "public-handler:removeLoyaltyPromotionsRoute": "Remove applied promotions",
                "public-handler:contactForm": "Contact form submission",
                "public-handler:cronHandler": "Scheduled job handler",
            }
            return descriptions.get(handler, "API endpoint")

        handler_counts["Description"] = handler_counts["Handler"].apply(get_handler_description)

        st.dataframe(
            handler_counts,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Handler": st.column_config.TextColumn("Route"),
                "Request Count": st.column_config.NumberColumn("Requests", format="%d"),
                "Description": st.column_config.TextColumn("Description"),
            },
        )


def render_operation_categories(df: pd.DataFrame):
    """Render operation categories breakdown with pie chart."""
    st.subheader("Operation Categories")

    if len(df) == 0:
        st.warning("No data available for the selected filters.")
        return

    # Extract category from operation name (first part before the dot)
    def get_category(name: str) -> str:
        if pd.isna(name):
            return "other"
        parts = name.split(".")
        if len(parts) >= 1:
            category = parts[0]
            # Normalize category names
            category_map = {
                "handlers": "Handlers",
                "middlewares": "Middlewares",
                "middleware": "Middlewares",
                "resolvers": "Resolvers",
                "utils": "Utilities",
                "request": "HTTP Requests",
            }
            return category_map.get(category, category.title())
        return "Other"

    df_with_category = df.copy()
    df_with_category["category"] = df_with_category["name"].apply(get_category)

    # Aggregate by category
    category_counts = df_with_category["category"].value_counts().reset_index()
    category_counts.columns = ["Category", "Count"]

    # Category descriptions
    category_info = {
        "Handlers": "Request handlers that process incoming API calls",
        "Middlewares": "Functions that run before/after request handlers",
        "Resolvers": "GraphQL field resolvers for data fetching",
        "Utilities": "Helper functions for common operations",
        "HTTP Requests": "Top-level HTTP request tracking",
    }

    col1, col2 = st.columns([1, 1])

    with col1:
        # Pie chart
        fig = px.pie(
            category_counts,
            values="Count",
            names="Category",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Category table with descriptions
        st.markdown("**Category Descriptions:**")
        for _, row in category_counts.iterrows():
            cat = row["Category"]
            count = row["Count"]
            desc = category_info.get(cat, "Other operations")
            st.markdown(f"- **{cat}** ({count:,}): {desc}")


def apply_filters(
    df: pd.DataFrame,
    time_range: tuple,
    clients: list,
    handlers: list,
    duration_threshold: int,
) -> pd.DataFrame:
    """Apply sidebar filters to the dataframe."""
    filtered = df.copy()

    # Time range filter
    filtered = filtered[
        (filtered["timestamp"] >= pd.Timestamp(time_range[0]))
        & (filtered["timestamp"] <= pd.Timestamp(time_range[1]))
    ]

    # Client filter
    if clients:
        filtered = filtered[filtered["client"].isin(clients)]

    # Handler filter
    if handlers:
        filtered = filtered[filtered["handler"].isin(handlers)]

    # Duration threshold filter
    if duration_threshold > 0:
        filtered = filtered[filtered["duration_ms"] >= duration_threshold]

    return filtered


def render_kpi_metrics(df: pd.DataFrame, time_range_minutes: float, is_requests: bool = True):
    """Render the KPI metric cards."""
    total_count = len(df)
    avg_duration = df["duration_ms"].mean() if len(df) > 0 else 0
    p50_duration = df["duration_ms"].quantile(0.50) if len(df) > 0 else 0
    p95_duration = df["duration_ms"].quantile(0.95) if len(df) > 0 else 0
    p99_duration = df["duration_ms"].quantile(0.99) if len(df) > 0 else 0
    slow_pct = (df["duration_ms"] > 1000).mean() * 100 if len(df) > 0 else 0
    throughput = total_count / time_range_minutes if time_range_minutes > 0 else 0

    # Calculate Apdex score with appropriate threshold
    apdex_threshold = 500 if is_requests else 100
    apdex_score = calculate_apdex(df, threshold_ms=apdex_threshold)
    apdex_color, apdex_label = get_apdex_color_and_label(apdex_score)

    # Use context-appropriate labels
    count_label = "HTTP Requests" if is_requests else "Total Operations"
    duration_label = "Avg Response Time" if is_requests else "Avg Duration"
    slow_label = "Slow Requests (>1s)" if is_requests else "Slow Operations (>1s)"
    throughput_label = "Throughput" if is_requests else "Ops/min"

    # Row 1: Core metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(count_label, f"{total_count:,}")
    with col2:
        st.metric(duration_label, f"{avg_duration:.0f} ms")
    with col3:
        st.metric(slow_label, f"{slow_pct:.1f}%")
    with col4:
        st.metric(throughput_label, f"{throughput:.0f}/min")
    with col5:
        # Apdex with colored label
        st.metric("Apdex Score", f"{apdex_score:.2f}")
        st.markdown(
            f"<p style='margin-top:-15px;color:{apdex_color};font-size:14px;'>{apdex_label} (T={apdex_threshold}ms)</p>",
            unsafe_allow_html=True,
        )

    # Row 2: Latency percentiles
    with st.expander("Latency Percentiles", expanded=True):
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            st.metric("P50 (Median)", f"{p50_duration:.0f} ms")
        with pcol2:
            st.metric("P95", f"{p95_duration:.0f} ms")
        with pcol3:
            st.metric("P99", f"{p99_duration:.0f} ms")


def render_time_series(df: pd.DataFrame):
    """Render time series chart with request volume and avg duration."""
    st.subheader("Request Volume & Duration Over Time")

    if len(df) == 0:
        st.warning("No data available for the selected filters.")
        return

    # Aggregate by minute
    df_ts = df.set_index("timestamp").resample("1min").agg(
        request_count=("duration_ms", "count"),
        avg_duration=("duration_ms", "mean"),
    ).reset_index()

    # Create dual-axis chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df_ts["timestamp"],
            y=df_ts["request_count"],
            name="Request Count",
            marker_color="steelblue",
            opacity=0.7,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df_ts["timestamp"],
            y=df_ts["avg_duration"],
            name="Avg Duration (ms)",
            line=dict(color="firebrick", width=2),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Request Count", secondary_y=False)
    fig.update_yaxes(title_text="Avg Duration (ms)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)


def render_duration_histogram(df: pd.DataFrame):
    """Render duration distribution histogram."""
    st.subheader("Duration Distribution")

    if len(df) == 0:
        st.warning("No data available for the selected filters.")
        return

    fig = px.histogram(
        df,
        x="duration_ms",
        nbins=50,
        log_y=True,
        labels={"duration_ms": "Duration (ms)", "count": "Count"},
        color_discrete_sequence=["steelblue"],
    )

    # Add threshold lines
    fig.add_vline(x=1000, line_dash="dash", line_color="orange", annotation_text="1s threshold")
    fig.add_vline(x=df["duration_ms"].median(), line_dash="dot", line_color="green", annotation_text="Median")

    fig.update_layout(
        height=350,
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_box_plot_by_handler(df: pd.DataFrame):
    """Render box plot of duration by handler."""
    st.subheader("Duration by Handler")

    if len(df) == 0:
        st.warning("No data available for the selected filters.")
        return

    # Get handler order by median duration
    handler_order = df.groupby("handler")["duration_ms"].median().sort_values(ascending=False).index.tolist()

    fig = px.box(
        df,
        x="handler",
        y="duration_ms",
        category_orders={"handler": handler_order},
        color_discrete_sequence=["steelblue"],
    )

    fig.update_layout(
        height=350,
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=40, b=80),
    )
    fig.update_yaxes(title_text="Duration (ms)")
    fig.update_xaxes(title_text="Handler")

    st.plotly_chart(fig, use_container_width=True)


def render_handler_table(df: pd.DataFrame):
    """Render handler comparison table with statistics."""
    st.subheader("Handler Performance Comparison")

    if len(df) == 0:
        st.warning("No data available for the selected filters.")
        return

    # Aggregate statistics by handler
    handler_stats = df.groupby("handler").agg(
        count=("duration_ms", "count"),
        avg_ms=("duration_ms", "mean"),
        p50_ms=("duration_ms", lambda x: x.quantile(0.50)),
        p95_ms=("duration_ms", lambda x: x.quantile(0.95)),
        p99_ms=("duration_ms", lambda x: x.quantile(0.99)),
    ).reset_index()

    # Round numeric columns
    for col in ["avg_ms", "p50_ms", "p95_ms", "p99_ms"]:
        handler_stats[col] = handler_stats[col].round(0).astype(int)

    # Sort by average duration descending
    handler_stats = handler_stats.sort_values("avg_ms", ascending=False)

    # Style function for duration columns
    def color_duration(val):
        if val > 3000:
            return "background-color: #6b2c2c"  # dark red
        elif val > 1000:
            return "background-color: #6b5a2c"  # dark amber
        elif val > 500:
            return "background-color: #4a5a2c"  # dark yellow-green
        return ""

    styled_df = handler_stats.style.map(
        color_duration, subset=["avg_ms", "p50_ms", "p95_ms", "p99_ms"]
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=min(400, 35 + 35 * len(handler_stats)),
        column_config={
            "handler": st.column_config.TextColumn("Handler"),
            "count": st.column_config.NumberColumn("Count", format="%d"),
            "avg_ms": st.column_config.NumberColumn("Avg (ms)", format="%d"),
            "p50_ms": st.column_config.NumberColumn("P50 (ms)", format="%d"),
            "p95_ms": st.column_config.NumberColumn("P95 (ms)", format="%d"),
            "p99_ms": st.column_config.NumberColumn("P99 (ms)", format="%d"),
        },
        hide_index=True,
    )


def render_box_plot_by_client(df: pd.DataFrame):
    """Render box plot of duration by client."""
    st.subheader("Duration by Client")

    if len(df) == 0:
        st.warning("No data available for the selected filters.")
        return

    # Get client order by median duration
    client_order = df.groupby("client")["duration_ms"].median().sort_values(ascending=False).index.tolist()

    fig = px.box(
        df,
        x="client",
        y="duration_ms",
        category_orders={"client": client_order},
        color_discrete_sequence=["steelblue"],
    )

    fig.update_layout(
        height=350,
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=40, b=80),
    )
    fig.update_yaxes(title_text="Duration (ms)")
    fig.update_xaxes(title_text="Client")

    st.plotly_chart(fig, use_container_width=True)


def render_slowest_operations(df: pd.DataFrame):
    """Render top 10 slowest operations bar chart."""
    st.subheader("Top 10 Slowest Operations (by Avg)")

    if len(df) == 0:
        st.warning("No data available for the selected filters.")
        return

    # Aggregate by operation name
    op_stats = df.groupby("name").agg(
        avg_duration=("duration_ms", "mean"),
        count=("duration_ms", "count"),
    ).reset_index()

    # Filter to operations with at least 10 requests and get top 10
    op_stats = op_stats[op_stats["count"] >= 10].nlargest(10, "avg_duration")

    fig = px.bar(
        op_stats,
        x="avg_duration",
        y="name",
        orientation="h",
        labels={"avg_duration": "Avg Duration (ms)", "name": "Operation"},
        color="avg_duration",
        color_continuous_scale="Reds",
        hover_data=["count"],
    )

    fig.update_layout(
        height=400,
        yaxis=dict(categoryorder="total ascending"),
        margin=dict(l=40, r=40, t=40, b=40),
        coloraxis_showscale=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_client_donut(df: pd.DataFrame):
    """Render request volume by client donut chart."""
    st.subheader("Request Volume by Client")

    if len(df) == 0:
        st.warning("No data available for the selected filters.")
        return

    client_counts = df["client"].value_counts().reset_index()
    client_counts.columns = ["client", "count"]

    fig = px.pie(
        client_counts,
        values="count",
        names="client",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )

    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_cumulative_time(df: pd.DataFrame):
    """Render cumulative time by operation chart and table."""
    st.subheader("Cumulative Time by Operation")

    if len(df) == 0:
        st.warning("No data available for the selected filters.")
        return

    # Aggregate by operation name
    op_stats = df.groupby("name").agg(
        count=("duration_ms", "count"),
        avg_ms=("duration_ms", "mean"),
    ).reset_index()

    # Calculate total time in seconds
    op_stats["total_time_s"] = (op_stats["count"] * op_stats["avg_ms"]) / 1000
    op_stats["avg_ms"] = op_stats["avg_ms"].round(0).astype(int)

    # Calculate percentage of total
    total_time = op_stats["total_time_s"].sum()
    op_stats["pct_of_total"] = (op_stats["total_time_s"] / total_time * 100).round(1)

    # Sort by total time descending
    op_stats = op_stats.sort_values("total_time_s", ascending=False)

    # Show top 15 operations in horizontal bar chart
    top_ops = op_stats.head(15)

    fig = px.bar(
        top_ops,
        x="total_time_s",
        y="name",
        orientation="h",
        labels={"total_time_s": "Total Time (s)", "name": "Operation"},
        color="total_time_s",
        color_continuous_scale="Blues",
        hover_data=["count", "avg_ms", "pct_of_total"],
    )

    fig.update_layout(
        height=450,
        yaxis=dict(categoryorder="total ascending"),
        margin=dict(l=40, r=40, t=40, b=40),
        coloraxis_showscale=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show table with all operations
    with st.expander("Full Operation Breakdown", expanded=False):
        st.dataframe(
            op_stats,
            use_container_width=True,
            height=400,
            column_config={
                "name": st.column_config.TextColumn("Operation"),
                "count": st.column_config.NumberColumn("Count", format="%d"),
                "avg_ms": st.column_config.NumberColumn("Avg (ms)", format="%d"),
                "total_time_s": st.column_config.NumberColumn("Total Time (s)", format="%.1f"),
                "pct_of_total": st.column_config.NumberColumn("% of Total", format="%.1f%%"),
            },
            hide_index=True,
        )


def render_data_table(df: pd.DataFrame):
    """Render filterable data table with conditional formatting."""
    st.subheader("Data Table")

    if len(df) == 0:
        st.warning("No data available for the selected filters.")
        return

    # Prepare display dataframe
    display_df = df[["timestamp", "name", "duration_ms", "client", "handler", "operation"]].copy()
    display_df = display_df.sort_values("duration_ms", ascending=False).head(1000)

    # Style the dataframe
    def color_duration(val):
        if val > 3000:
            return "background-color: #6b2c2c"  # dark red
        elif val > 1000:
            return "background-color: #6b5a2c"  # dark amber
        return ""

    styled_df = display_df.style.map(color_duration, subset=["duration_ms"])

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
            "name": st.column_config.TextColumn("Operation Name"),
            "duration_ms": st.column_config.NumberColumn("Duration (ms)", format="%d"),
            "client": st.column_config.TextColumn("Client"),
            "handler": st.column_config.TextColumn("Handler"),
            "operation": st.column_config.TextColumn("Operation"),
        },
    )


def main():
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="Salesforce Integration Log Analytics",
        page_icon="📊",
        layout="wide",
    )

    st.title("Salesforce Integration Log Analytics")

    # Load data
    with st.spinner("Loading data..."):
        df = load_data()

    if len(df) == 0:
        st.error("No data found in the database.")
        return

    # Render data overview at the top of sidebar
    render_data_overview(df)

    filter_options = get_filter_options(df)

    # Sidebar filters
    st.sidebar.header("Filters")

    min_time = df["timestamp"].min().to_pydatetime()
    max_time = df["timestamp"].max().to_pydatetime()

    time_range = st.sidebar.slider(
        "Time Range",
        min_value=min_time,
        max_value=max_time,
        value=(min_time, max_time),
        format="HH:mm:ss",
    )

    selected_clients = st.sidebar.multiselect(
        "Clients",
        options=filter_options["clients"],
        default=[],
        placeholder="All clients",
    )

    selected_handlers = st.sidebar.multiselect(
        "Handlers",
        options=filter_options["handlers"],
        default=[],
        placeholder="All handlers",
    )

    duration_threshold = st.sidebar.slider(
        "Min Duration (ms)",
        min_value=0,
        max_value=5000,
        value=0,
        step=100,
    )

    # Advanced Statistics Settings
    st.sidebar.markdown("---")
    st.sidebar.header("Advanced Statistics")

    sla_threshold = st.sidebar.slider(
        "SLA Threshold (ms)",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Requests/operations under this threshold are considered SLA-compliant",
    )

    slow_request_threshold = st.sidebar.slider(
        "Slow Request Threshold (ms)",
        min_value=500,
        max_value=5000,
        value=1000,
        step=100,
        help="Requests exceeding this threshold are analyzed in the breakdown",
    )

    enable_period_comparison = st.sidebar.checkbox(
        "Enable Period Comparison",
        value=False,
        help="Compare current time selection against previous equivalent period",
    )

    # Apply filters
    filtered_df = apply_filters(
        df,
        time_range,
        selected_clients,
        selected_handlers,
        duration_threshold,
    )

    # Calculate time range in minutes
    time_range_minutes = (time_range[1] - time_range[0]).total_seconds() / 60

    # Split data into HTTP requests and operations
    requests_df = filtered_df[filtered_df["name"] == "request"]
    operations_df = filtered_df[filtered_df["name"] != "request"]

    # Show filter summary with counts for both categories
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Filtered Records: {len(filtered_df):,} of {len(df):,}**")
    st.sidebar.markdown(f"- HTTP Requests: {len(requests_df):,}")
    st.sidebar.markdown(f"- Operations: {len(operations_df):,}")

    # Render glossary/help section
    st.sidebar.markdown("---")
    render_glossary()

    # Create tabs for HTTP Requests and Operations
    tab_requests, tab_operations = st.tabs(["HTTP Requests", "Operations"])

    # Calculate period comparison if enabled
    requests_comparison = None
    operations_comparison = None
    if enable_period_comparison:
        requests_comparison = calculate_period_comparison(
            df[df["name"] == "request"],
            pd.Timestamp(time_range[0]),
            pd.Timestamp(time_range[1]),
        )
        operations_comparison = calculate_period_comparison(
            df[df["name"] != "request"],
            pd.Timestamp(time_range[0]),
            pd.Timestamp(time_range[1]),
        )

    with tab_requests:
        # Route reference (collapsible)
        render_handler_reference(requests_df)

        # KPI metrics for HTTP requests
        render_kpi_metrics(requests_df, time_range_minutes, is_requests=True)

        # Period comparison (if enabled)
        if enable_period_comparison:
            st.markdown("---")
            render_period_comparison(requests_comparison, is_requests=True)

        st.markdown("---")

        # Ops per Request Analysis (NEW)
        render_ops_per_request(requests_df, operations_df)

        st.markdown("---")

        # Throughput Analysis (NEW - enhanced)
        render_throughput_analysis(requests_df, time_range_minutes, is_requests=True)

        st.markdown("---")

        # Time Series (original)
        render_time_series(requests_df)

        st.markdown("---")

        # Distribution (2 columns)
        col1, col2 = st.columns(2)
        with col1:
            render_duration_histogram(requests_df)
        with col2:
            render_box_plot_by_handler(requests_df)

        st.markdown("---")

        # Handler performance comparison table
        render_handler_table(requests_df)

        st.markdown("---")

        # Slow Request Breakdown (NEW)
        render_slow_request_breakdown(requests_df, operations_df, slow_request_threshold)

        st.markdown("---")

        # Data Table
        render_data_table(requests_df)

    with tab_operations:
        # Operation categories breakdown
        render_operation_categories(operations_df)

        st.markdown("---")

        # KPI metrics for operations
        render_kpi_metrics(operations_df, time_range_minutes, is_requests=False)

        # Period comparison (if enabled)
        if enable_period_comparison:
            st.markdown("---")
            render_period_comparison(operations_comparison, is_requests=False)

        st.markdown("---")

        # Throughput Analysis (NEW - enhanced)
        render_throughput_analysis(operations_df, time_range_minutes, is_requests=False)

        st.markdown("---")

        # Time Series
        render_time_series(operations_df)

        st.markdown("---")

        # Distribution (2 columns)
        col1, col2 = st.columns(2)
        with col1:
            render_duration_histogram(operations_df)
        with col2:
            render_slowest_operations(operations_df)

        st.markdown("---")

        # Cumulative time by operation (optimization targets)
        render_cumulative_time(operations_df)

        st.markdown("---")

        # Client SLA Compliance (NEW)
        render_sla_compliance(operations_df, sla_threshold, is_requests=False)

        st.markdown("---")

        # Breakdown (2 columns)
        col1, col2 = st.columns(2)
        with col1:
            render_box_plot_by_client(operations_df)
        with col2:
            render_client_donut(operations_df)

        st.markdown("---")

        # Data Table
        render_data_table(operations_df)


if __name__ == "__main__":
    main()
