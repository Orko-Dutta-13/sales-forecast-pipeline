import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import duckdb
from datetime import datetime, timedelta


# ============================================================
# PAGE CONFIGURATION
# ============================================================

# Must be the first Streamlit command
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

BASE_URL = "http://127.0.0.1:8000"

DB_PATH = r"C:\Users\oorko\OneDrive\Documents\projects\Sales_Forecasting\data\sales_forecast.db"


# ============================================================
# LOAD REFERENCE DATA
# ============================================================

@st.cache_data
def load_reference_data():
    """
    Load store list and family list once and cache it.
    Cached data persists across user interactions so DuckDB
    isn't queried every time someone moves a slider.
    """

    conn = duckdb.connect(DB_PATH)

    # raw_stores contains a column called "type",
    # so rename it to "store_type" for the dashboard
    stores = conn.execute("""
        SELECT
            store_nbr,
            type AS store_type,
            city,
            state,
            cluster
        FROM raw_stores
        ORDER BY store_nbr
    """).fetchdf()

    families = conn.execute("""
        SELECT DISTINCT family
        FROM raw_train
        ORDER BY family
    """).fetchdf()

    conn.close()

    return stores, families


stores_df, families_df = load_reference_data()


# ============================================================
# LOAD HISTORICAL SALES
# ============================================================

@st.cache_data
def load_historical_sales(store_nbr, family, days=90):
    """
    Pull last N days of actual sales for the selected store + family.
    Also pulls the last 28 days needed to compute lag features
    for forecasting.
    """

    conn = duckdb.connect(DB_PATH)

    df = conn.execute(f"""
        SELECT
            date,
            sales,
            onpromotion,
            oil_price,
            is_national_holiday,
            transactions,
            sales_lag_7,
            sales_lag_14,
            sales_lag_28,
            sales_rolling_7day_avg,
            sales_rolling_28day_avg
        FROM mart_sales_features
        WHERE store_nbr = {store_nbr}
          AND family = '{family}'
          AND sales_lag_7 IS NOT NULL
        ORDER BY date DESC
        LIMIT {days + 28}
    """).fetchdf()

    conn.close()

    df = df.sort_values("date").reset_index(drop=True)

    return df


# ============================================================
# GENERATE FORECAST
# ============================================================

def generate_forecast(
    store_nbr,
    family,
    historical_df,
    horizon_days=7
):
    """
    Generate predictions for the next N days by calling
    the FastAPI prediction endpoint.

    Uses historical sales to build lag and rolling features.
    """

    # Last date available in dataset
    last_date = historical_df["date"].max()

    # Use most recent available oil price
    last_oil = historical_df["oil_price"].iloc[-1]

    forecast_rows = []

    for i in range(1, horizon_days + 1):

        forecast_date = last_date + timedelta(days=i)

        # ----------------------------------------------------
        # Lag dates
        # ----------------------------------------------------

        lag7_date = forecast_date - timedelta(days=7)
        lag14_date = forecast_date - timedelta(days=14)
        lag28_date = forecast_date - timedelta(days=28)


        # ----------------------------------------------------
        # Find sales value on a particular historical date
        # ----------------------------------------------------

        def get_sales_on(target_date):

            row = historical_df[
                historical_df["date"] == pd.Timestamp(target_date)
            ]

            if len(row) > 0:
                return float(row["sales"].values[0])

            # Fallback if date not available
            return float(
                historical_df["sales"].iloc[-1]
            )


        lag7 = get_sales_on(lag7_date)
        lag14 = get_sales_on(lag14_date)
        lag28 = get_sales_on(lag28_date)


        # ----------------------------------------------------
        # Rolling averages
        # ----------------------------------------------------

        rolling_7 = float(
            historical_df["sales"]
            .iloc[-7:]
            .mean()
        )

        rolling_28 = float(
            historical_df["sales"]
            .iloc[-28:]
            .mean()
        )


        # ----------------------------------------------------
        # API request payload
        # ----------------------------------------------------

        payload = {

            "store_nbr":
                store_nbr,

            "family":
                family,

            "date":
                str(forecast_date.date()),

            "onpromotion":
                int(
                    historical_df["onpromotion"].iloc[-1]
                ),

            "oil_price":
                float(last_oil),

            "is_national_holiday":
                0,

            "transactions":
                float(
                    historical_df["transactions"].iloc[-1]
                ),

            "sales_lag_7":
                lag7,

            "sales_lag_14":
                lag14,

            "sales_lag_28":
                lag28,

            "sales_rolling_7day_avg":
                rolling_7,

            "sales_rolling_28day_avg":
                rolling_28
        }


        # ----------------------------------------------------
        # Call FastAPI
        # ----------------------------------------------------

        try:

            response = requests.post(
                f"{BASE_URL}/predict",
                json=payload,
                timeout=5
            )

            if response.status_code == 200:

                predicted = response.json()[
                    "predicted_sales"
                ]

            else:

                predicted = None

        except Exception:

            predicted = None


        # ----------------------------------------------------
        # Save forecast row
        # ----------------------------------------------------

        forecast_rows.append({

            "date":
                forecast_date,

            "predicted_sales":
                predicted
        })


    return pd.DataFrame(forecast_rows)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔧 Filters")

st.sidebar.markdown("---")


# ------------------------------------------------------------
# Store selection
# ------------------------------------------------------------

selected_store = st.sidebar.selectbox(

    "Select Store",

    options=
        stores_df["store_nbr"].tolist(),

    format_func=
        lambda x:
        f"Store {x} "
        f"({stores_df[stores_df['store_nbr'] == x]['city'].values[0]})"
)


# ------------------------------------------------------------
# Product family selection
# ------------------------------------------------------------

selected_family = st.sidebar.selectbox(

    "Select Product Family",

    options=
        families_df["family"].tolist(),

    index=
        families_df["family"]
        .tolist()
        .index("GROCERY I")
)


# ------------------------------------------------------------
# Forecast horizon
# ------------------------------------------------------------

forecast_horizon = st.sidebar.slider(

    "Forecast Horizon (days)",

    min_value=1,

    max_value=7,

    value=7
)


# ------------------------------------------------------------
# Historical data slider
# ------------------------------------------------------------

history_days = st.sidebar.slider(

    "Historical Days to Show",

    min_value=30,

    max_value=90,

    value=60
)


st.sidebar.markdown("---")


# ============================================================
# STORE INFORMATION
# ============================================================

store_info = stores_df[
    stores_df["store_nbr"] == selected_store
].iloc[0]


st.sidebar.markdown(
    f"""
**Store {selected_store} Details**

- City: {store_info['city']}
- State: {store_info['state']}
- Type: {store_info['store_type']}
- Cluster: {store_info['cluster']}
"""
)


# ============================================================
# MAIN CONTENT
# ============================================================

st.title(
    "📈 Retail Sales Forecast Dashboard"
)


st.markdown(
    f"""
**Store {selected_store}**
|
**{selected_family}**
|
Forecasting {forecast_horizon} days ahead
"""
)


st.markdown("---")


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

with st.spinner(
    "Loading sales data..."
):

    historical_df = load_historical_sales(

        selected_store,

        selected_family,

        days=history_days
    )


# ============================================================
# CHECK HISTORICAL DATA
# ============================================================

if historical_df.empty:

    st.error(
        "No historical sales data found "
        "for this store and product family."
    )

    st.stop()


# ============================================================
# CHECK FASTAPI
# ============================================================

try:

    api_check = requests.get(
        f"{BASE_URL}/",
        timeout=3
    )

    api_running = (
        api_check.status_code == 200
    )

except Exception:

    api_running = False


if not api_running:

    st.error(
        "⚠️ FastAPI is not running. "
        "Start it with: uvicorn main:app --reload"
    )

    st.stop()


# ============================================================
# GENERATE FORECAST
# ============================================================

with st.spinner(
    f"Generating {forecast_horizon}-day forecast..."
):

    forecast_df = generate_forecast(

        selected_store,

        selected_family,

        historical_df,

        forecast_horizon
    )


# ============================================================
# CHECK FORECAST
# ============================================================

if forecast_df["predicted_sales"].isna().all():

    st.error(
        "Forecast could not be generated. "
        "Check the FastAPI terminal for an error."
    )

    st.stop()


# ============================================================
# KPI CARDS
# ============================================================

st.subheader(
    "📊 Forecast Summary"
)


col1, col2, col3, col4 = st.columns(4)


total_forecast = (
    forecast_df["predicted_sales"].sum()
)


avg_daily = (
    forecast_df["predicted_sales"].mean()
)


best_day_val = (
    forecast_df["predicted_sales"].max()
)


best_day_index = (
    forecast_df["predicted_sales"].idxmax()
)


best_day_date = (
    forecast_df
    .loc[best_day_index, "date"]
    .strftime("%b %d")
)


last_actual = (
    historical_df["sales"].iloc[-1]
)


if last_actual > 0:

    pct_change = (
        (avg_daily - last_actual)
        / last_actual
    ) * 100

else:

    pct_change = 0


# ------------------------------------------------------------
# KPI 1
# ------------------------------------------------------------

col1.metric(

    label=
        "Total Predicted Sales",

    value=
        f"{total_forecast:,.0f}",

    delta=
        f"{forecast_horizon} day total"
)


# ------------------------------------------------------------
# KPI 2
# ------------------------------------------------------------

col2.metric(

    label=
        "Avg Daily Sales",

    value=
        f"{avg_daily:,.0f}",

    delta=
        f"{pct_change:+.1f}% vs last actual"
)


# ------------------------------------------------------------
# KPI 3
# ------------------------------------------------------------

col3.metric(

    label=
        "Best Forecast Day",

    value=
        f"{best_day_val:,.0f}",

    delta=
        best_day_date
)


# ------------------------------------------------------------
# KPI 4
# ------------------------------------------------------------

col4.metric(

    label=
        "Last Actual Sale",

    value=
        f"{last_actual:,.0f}",

    delta=
        "Most recent data point"
)


# ============================================================
# MAIN HISTORICAL + FORECAST CHART
# ============================================================

st.markdown("---")

st.subheader(
    "📉 Historical Sales + Forecast"
)


display_historical = (
    historical_df.tail(history_days)
)


fig = go.Figure()


# ------------------------------------------------------------
# Historical sales
# ------------------------------------------------------------

fig.add_trace(

    go.Scatter(

        x=
            display_historical["date"],

        y=
            display_historical["sales"],

        mode=
            "lines",

        name=
            "Actual Sales",

        line=
            dict(
                color="#2196F3",
                width=2
            ),

        hovertemplate=
            "Date: %{x}<br>"
            "Actual Sales: %{y:,.0f}"
            "<extra></extra>"
    )
)


# ------------------------------------------------------------
# Forecast
# ------------------------------------------------------------

fig.add_trace(

    go.Scatter(

        x=
            forecast_df["date"],

        y=
            forecast_df["predicted_sales"],

        mode=
            "lines+markers",

        name=
            "Forecast",

        line=
            dict(
                color="#FF5722",
                width=2,
                dash="dash"
            ),

        marker=
            dict(
                size=8,
                symbol="circle"
            ),

        hovertemplate=
            "Date: %{x}<br>"
            "Predicted: %{y:,.0f}"
            "<extra></extra>"
    )
)


# ------------------------------------------------------------
# Forecast shaded region
# ------------------------------------------------------------

fig.add_vrect(

    x0=
        forecast_df["date"].min(),

    x1=
        forecast_df["date"].max(),

    fillcolor=
        "rgba(255, 87, 34, 0.08)",

    layer=
        "below",

    line_width=
        0,

    annotation_text=
        "Forecast Period",

    annotation_position=
        "top left"
)


# ------------------------------------------------------------
# Forecast start line
# ------------------------------------------------------------

fig.add_vline(

    x=
        historical_df["date"].max(),

    line_dash=
        "dot",

    line_color=
        "gray",

    annotation_text=
        "Forecast Start",

    annotation_position=
        "top right"
)


# ------------------------------------------------------------
# Chart layout
# ------------------------------------------------------------

fig.update_layout(

    xaxis_title=
        "Date",

    yaxis_title=
        "Sales",

    hovermode=
        "x unified",

    legend=
        dict(
            orientation="h",
            yanchor="bottom",
            y=1.02
        ),

    height=
        450,

    template=
        "plotly_white"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# DAILY FORECAST BAR CHART
# ============================================================

st.subheader(
    "📅 Day-by-Day Forecast Breakdown"
)


fig2 = go.Figure()


colors = [

    "#FF5722"
    if i == best_day_index

    else "#FF8A65"

    for i in forecast_df.index
]


fig2.add_trace(

    go.Bar(

        x=
            forecast_df["date"]
            .dt.strftime("%a %b %d"),

        y=
            forecast_df["predicted_sales"],

        marker_color=
            colors,

        text=
            forecast_df[
                "predicted_sales"
            ].apply(
                lambda x:
                f"{x:,.0f}"
            ),

        textposition=
            "outside",

        hovertemplate=
            "%{x}<br>"
            "Predicted: %{y:,.0f}"
            "<extra></extra>"
    )
)


fig2.update_layout(

    xaxis_title=
        "Date",

    yaxis_title=
        "Predicted Sales",

    height=
        350,

    template=
        "plotly_white",

    showlegend=
        False
)


st.plotly_chart(
    fig2,
    use_container_width=True
)


# ============================================================
# DATA TABLES
# ============================================================

st.markdown("---")


col_left, col_right = st.columns(2)


# ------------------------------------------------------------
# Forecast table
# ------------------------------------------------------------

with col_left:

    st.subheader(
        "🔮 Forecast Data"
    )

    forecast_display = (
        forecast_df.copy()
    )

    forecast_display["date"] = (
        forecast_display["date"]
        .dt.strftime("%Y-%m-%d")
    )

    forecast_display[
        "predicted_sales"
    ] = (
        forecast_display[
            "predicted_sales"
        ].round(2)
    )

    forecast_display.columns = [
        "Date",
        "Predicted Sales"
    ]

    st.dataframe(

        forecast_display,

        use_container_width=True,

        hide_index=True
    )


# ------------------------------------------------------------
# Recent actual sales table
# ------------------------------------------------------------

with col_right:

    st.subheader(
        "📋 Recent Actual Sales"
    )

    actual_display = (
        historical_df[
            ["date", "sales"]
        ]
        .tail(forecast_horizon)
        .copy()
    )

    actual_display["date"] = (
        actual_display["date"]
        .dt.strftime("%Y-%m-%d")
    )

    actual_display["sales"] = (
        actual_display["sales"]
        .round(2)
    )

    actual_display.columns = [
        "Date",
        "Actual Sales"
    ]

    st.dataframe(

        actual_display,

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")


st.markdown(
    """
<div style='text-align: center; color: gray; font-size: 0.85em;'>

Built with XGBoost + FastAPI + Streamlit |
Data: Kaggle Store Sales - Time Series Forecasting |
Model RMSLE: 0.5468

</div>
""",
    unsafe_allow_html=True
)