

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

import joblib
import requests
import os

# ============================================================
# RANDOM FOREST MODEL
# ============================================================

MODEL_PATH = "random_forest_model_compressed.joblib"

MODEL_URL = "https://huggingface.co/DhanashriKatkar/weather-temperature-random-forest/resolve/main/random_forest_model_compressed.joblib"

# Download model if it is not already available locally
if not os.path.exists(MODEL_PATH):

    response = requests.get(MODEL_URL)

    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

    print("Random Forest model downloaded successfully.")


# Load model
rf_model = joblib.load(MODEL_PATH)

print("Random Forest model loaded successfully.")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Indian Weather Analysis",
    page_icon="🌤️",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

weather = pd.read_csv(
    r"Dataset//Indian_Weather_Consolidated.zip"
)


# ============================================================
# DATA PREPARATION
# ============================================================

weather["date_time"] = pd.to_datetime(
    weather["date_time"],
    errors="coerce"
)

# Year
if "year" not in weather.columns:
    weather["year"] = weather["date_time"].dt.year

# Month number
if "month_num" not in weather.columns:
    weather["month_num"] = weather["date_time"].dt.month

# Month name
if "month" not in weather.columns:
    weather["month"] = weather["date_time"].dt.month_name()

# Date
if "date" not in weather.columns:
    weather["date"] = weather["date_time"].dt.date


# Make sure year is numeric
weather["year"] = pd.to_numeric(
    weather["year"],
    errors="coerce"
)


# ============================================================
# MONTH INFORMATION
# ============================================================

month_names = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]

month_short = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec"
]

month_map = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌤️ Weather Analysis")

st.sidebar.markdown(
    "### Dashboard Navigation"
)

selected_page = st.sidebar.radio(
    "Select Page",
    [
        "Temperature & Heat",
        "UV, Sunshine & Rainfall",
        "Wind, Visibility & Relationships"
        "Temperature Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    ### Dataset

    **Period:** 2009–2019

    **Cities:** 8 Indian cities

    **Analysis:**
    Temperature, Heat, UV,
    Sunshine, Rainfall,
    Wind & Visibility
    """
)


# ============================================================
# PAGE 1
# TEMPERATURE & HEAT
# ============================================================

if selected_page == "Temperature & Heat":

    st.title("🌡️ Temperature & Heat Analysis")

    st.markdown(
        """
        Analysis of temperature patterns, seasonal variation,
        heat index and humidity relationships across Indian cities
        from **2009–2019**.
        """
    )


        # ========================================================
    # 1. TEMPERATURE COMPARISON BY CITY
    # ========================================================

    st.header("1. Temperature Comparison by City")

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    temp_data = weather.copy()

    # Make sure year is numeric
    temp_data["year"] = pd.to_numeric(
        temp_data["year"],
        errors="coerce"
    )

    # Make sure temperature columns are numeric
    temp_columns = [
        "tempC",
        "mintempC",
        "maxtempC"
    ]

    for col in temp_columns:
        temp_data[col] = pd.to_numeric(
            temp_data[col],
            errors="coerce"
        )

    # Filter 2009–2019
    temp_data = temp_data[
        temp_data["year"].between(2009, 2019)
    ].copy()

    # Remove rows without city
    temp_data = temp_data.dropna(
        subset=["city"]
    )

    # --------------------------------------------------------
    # City-level temperature KPIs
    # --------------------------------------------------------

    city_temp = (
        temp_data
        .groupby("city", as_index=False)
        .agg(
            Avg_Temp=("tempC", "mean"),
            Min_Temp=("mintempC", "min"),
            Max_Temp=("maxtempC", "max")
        )
    )

    # Remove cities where all temperature values are missing
    city_temp = city_temp.dropna(
        subset=[
            "Avg_Temp",
            "Min_Temp",
            "Max_Temp"
        ],
        how="all"
    )

    # Sort by average temperature
    city_temp = city_temp.sort_values(
        "Avg_Temp"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Convert to long format
    # This makes Plotly more reliable
    # --------------------------------------------------------

    city_temp_long = city_temp.melt(
        id_vars="city",
        value_vars=[
            "Min_Temp",
            "Avg_Temp",
            "Max_Temp"
        ],
        var_name="Temperature_Type",
        value_name="Temperature"
    )

    # --------------------------------------------------------
    # Rename labels for visual
    # --------------------------------------------------------

    city_temp_long["Temperature_Type"] = (
        city_temp_long["Temperature_Type"]
        .replace({
            "Min_Temp": "Minimum Temperature",
            "Avg_Temp": "Average Temperature",
            "Max_Temp": "Maximum Temperature"
        })
    )

    # --------------------------------------------------------
    # Plotly chart
    # --------------------------------------------------------

    fig = px.bar(
        city_temp_long,
        x="city",
        y="Temperature",
        color="Temperature_Type",
        barmode="group",

        title="Temperature Comparison by City (2009–2019)",

        labels={
            "city": "City",
            "Temperature": "Temperature (°C)",
            "Temperature_Type": "Temperature Type"
        },

        hover_data={
            "city": True,
            "Temperature_Type": True,
            "Temperature": ":.2f"
        }
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_layout(
        height=550,

        xaxis=dict(
            title="City",
            tickangle=-45,
            categoryorder="array",
            categoryarray=city_temp["city"].tolist()
        ),

        yaxis=dict(
            title="Temperature (°C)",
            rangemode="tozero"
        ),

        hovermode="x unified",

        legend=dict(
            title="Temperature Type"
        ),

        margin=dict(
            l=60,
            r=30,
            t=80,
            b=100
        )
    )

    fig.update_traces(
        hovertemplate=
        "<b>%{x}</b><br>"
        "Temperature Type: %{fullData.name}<br>"
        "Temperature: %{y:.2f} °C"
        "<extra></extra>"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # KPI calculations
    # --------------------------------------------------------

    if not city_temp.empty:

        highest_avg_city = city_temp.loc[
            city_temp["Avg_Temp"].idxmax()
        ]

        lowest_avg_city = city_temp.loc[
            city_temp["Avg_Temp"].idxmin()
        ]

        highest_extreme_city = city_temp.loc[
            city_temp["Max_Temp"].idxmax()
        ]

        # ----------------------------------------------------
        # KPI cards
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Highest Average Temperature",
                f"{highest_avg_city['Avg_Temp']:.2f} °C",
                highest_avg_city["city"]
            )

        with col2:
            st.metric(
                "Lowest Average Temperature",
                f"{lowest_avg_city['Avg_Temp']:.2f} °C",
                lowest_avg_city["city"]
            )

        with col3:
            st.metric(
                "Highest Extreme Temperature",
                f"{highest_extreme_city['Max_Temp']:.0f} °C",
                highest_extreme_city["city"]
            )

        # ----------------------------------------------------
        # Key Insights
        # ----------------------------------------------------

        st.info(
            f"""
            **Key Insights**

            • **{highest_avg_city['city']}** has the highest average
            temperature at approximately
            **{highest_avg_city['Avg_Temp']:.2f}°C**.

            • **{lowest_avg_city['city']}** has the lowest average
            temperature at approximately
            **{lowest_avg_city['Avg_Temp']:.2f}°C**.

            • **{highest_extreme_city['city']}** records the highest
            extreme temperature at approximately
            **{highest_extreme_city['Max_Temp']:.0f}°C**.

            • The comparison shows differences in average,
            minimum and maximum temperature across the eight cities.
            """
        )

    else:

        st.warning(
            "No temperature data is available for the selected period."
        )

    # ========================================================
    # 2. HIGHEST RECORDED TEMPERATURE
    # ========================================================

    st.header("2. Highest Recorded Temperature by City")

    city_max_temp = (
        weather[
            weather["year"].between(2009, 2019)
        ]
        .groupby("city")["maxtempC"]
        .max()
        .sort_values()
        .reset_index()
    )

    fig = px.bar(
        city_max_temp,
        x="maxtempC",
        y="city",
        orientation="h",
        title="Maximum Recorded Temperature by City",
        labels={
            "maxtempC": "Maximum Temperature (°C)",
            "city": "City"
        },
        text="maxtempC"
    )

    fig.update_traces(
        texttemplate="%{text:.0f}°C",
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Maximum Temperature: %{x:.1f}°C"
            "<extra></extra>"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    highest_temp = city_max_temp.loc[
        city_max_temp["maxtempC"].idxmax()
    ]

    st.info(
        f"""
        **Key Insights**

        • **{highest_temp['city']}** recorded the highest temperature
        at **{highest_temp['maxtempC']:.0f}°C**.

        • Inland cities show substantially higher temperature extremes
        than Bombay.
        """
    )


    # ========================================================
    # 3. MONTHLY TEMPERATURE TREND
    # ========================================================

    st.header("3. Monthly Temperature Trend")

    monthly_temp = (
        weather[
            weather["year"].between(2009, 2019)
        ]
        .groupby("month_num")["tempC"]
        .mean()
        .sort_index()
        .reset_index()
    )

    monthly_temp["Month"] = (
        monthly_temp["month_num"].map(month_map)
    )

    fig = px.line(
        monthly_temp,
        x="Month",
        y="tempC",
        markers=True,
        title="Monthly Temperature Trend (2009–2019)",
        labels={
            "tempC": "Average Temperature (°C)",
            "Month": "Month"
        },
        hover_data={
            "tempC": ":.2f"
        }
    )

    fig.update_layout(
        xaxis=dict(
            categoryorder="array",
            categoryarray=month_names
        ),
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    hottest_month = monthly_temp.loc[
        monthly_temp["tempC"].idxmax()
    ]

    coolest_month = monthly_temp.loc[
        monthly_temp["tempC"].idxmin()
    ]

    st.info(
        f"""
        **Key Insights**

        • **{hottest_month['Month']}** is the hottest month with an average
        temperature of approximately **{hottest_month['tempC']:.2f}°C**.

        • **{coolest_month['Month']}** is the coolest month with an average
        temperature of approximately **{coolest_month['tempC']:.2f}°C**.

        • Temperature rises sharply from winter toward summer.

        • Temperatures generally decline after the summer period.
        """
    )


    # ========================================================
    # 4. CITY × MONTH TEMPERATURE HEATMAP
    # ========================================================

    st.header("4. Average Temperature by City and Month")

    city_month_temp = (
        weather[
            weather["year"].between(2009, 2019)
        ]
        .groupby(
            ["city", "month_num"]
        )["tempC"]
        .mean()
        .reset_index()
    )

    heatmap_data = city_month_temp.pivot(
        index="city",
        columns="month_num",
        values="tempC"
    )

    heatmap_data = heatmap_data.reindex(
        columns=range(1, 13)
    )

    heatmap_data.columns = month_short

    fig = px.imshow(
        heatmap_data,
        text_auto=".1f",
        aspect="auto",
        title="Average Temperature by City and Month (2009–2019)",
        labels={
            "x": "Month",
            "y": "City",
            "color": "Temperature (°C)"
        }
    )

    fig.update_traces(
        hovertemplate=(
            "City: %{y}<br>"
            "Month: %{x}<br>"
            "Average Temperature: %{z:.2f}°C"
            "<extra></extra>"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        """
        **Key Insights**

        • **Delhi, Jaipur, Kanpur and Nagpur** show strong seasonal
        variation.

        • **Hyderabad and Pune** show moderate seasonal variation.

        • **Bengaluru and Bombay** have comparatively stable temperatures.

        • Nagpur records one of the highest city-month average temperatures.

        • Bombay has the smallest monthly temperature range among
        the analyzed cities.
        """
    )


    # ========================================================
    # 5. SEASONAL TEMPERATURE
    # ========================================================

    st.header("5. Average Temperature by Season")

    season_temp = (
        weather[
            weather["year"].between(2009, 2019)
        ]
        .groupby(
            "season",
            observed=True
        )["tempC"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        season_temp,
        x="season",
        y="tempC",
        title="Average Temperature by Season",
        labels={
            "season": "Season",
            "tempC": "Average Temperature (°C)"
        },
        text="tempC"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}°C",
        textposition="outside",
        hovertemplate=(
            "Season: %{x}<br>"
            "Average Temperature: %{y:.2f}°C"
            "<extra></extra>"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    hottest_season = season_temp.loc[
        season_temp["tempC"].idxmax()
    ]

    coolest_season = season_temp.loc[
        season_temp["tempC"].idxmin()
    ]

    st.info(
        f"""
        **Key Insights**

        • **{hottest_season['season']}** is the hottest season at
        approximately **{hottest_season['tempC']:.2f}°C**.

        • **{coolest_season['season']}** is the coolest season at
        approximately **{coolest_season['tempC']:.2f}°C**.

        • Seasonal temperature difference is approximately
        **{hottest_season['tempC'] - coolest_season['tempC']:.2f}°C**.
        """
    )


    # ========================================================
    # 6. HEAT INDEX BY CITY
    # ========================================================

    st.header("6. Average Heat Index by City")

    city_heat_index = (
        weather[
            weather["year"].between(2009, 2019)
        ]
        .groupby("city")["HeatIndexC"]
        .mean()
        .sort_values()
        .reset_index()
    )

    fig = px.bar(
        city_heat_index,
        x="HeatIndexC",
        y="city",
        orientation="h",
        title="Average Heat Index by City (2009–2019)",
        labels={
            "HeatIndexC": "Average Heat Index (°C)",
            "city": "City"
        },
        text="HeatIndexC"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}°C",
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Average Heat Index: %{x:.2f}°C"
            "<extra></extra>"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    highest_heat_city = city_heat_index.loc[
        city_heat_index["HeatIndexC"].idxmax()
    ]

    lowest_heat_city = city_heat_index.loc[
        city_heat_index["HeatIndexC"].idxmin()
    ]

    st.info(
        f"""
        **Key Insights**

        • **{highest_heat_city['city']}** has the highest average Heat Index
        at approximately **{highest_heat_city['HeatIndexC']:.2f}°C**.

        • **{lowest_heat_city['city']}** has the lowest average Heat Index
        at approximately **{lowest_heat_city['HeatIndexC']:.2f}°C**.

        • Air temperature alone does not fully represent perceived heat.
        """
    )


    # ========================================================
    # 7. MONTHLY TEMPERATURE VS HEAT INDEX
    # ========================================================

    st.header("7. Monthly Temperature vs Heat Index")

    st.markdown(
        "Select a city and analysis period for detailed heat-stress analysis."
    )

    cities = sorted(
        weather["city"].dropna().unique()
    )

    years = sorted(
        weather["year"].dropna().astype(int).unique()
    )

    selected_city = st.selectbox(
        "Select City",
        cities,
        key="heat_city"
    )

    analysis_type = st.radio(
        "Analysis Period",
        [
            "Specific Year",
            "Year Range"
        ],
        horizontal=True,
        key="heat_period"
    )

    if analysis_type == "Specific Year":

        selected_year = st.selectbox(
            "Select Year",
            years,
            key="heat_year"
        )

        data = weather[
            (weather["city"] == selected_city) &
            (weather["year"] == selected_year)
        ].copy()

        period_text = str(selected_year)

    else:

        col1, col2 = st.columns(2)

        with col1:
            start_year = st.selectbox(
                "Start Year",
                years,
                index=0,
                key="heat_start_year"
            )

        with col2:
            end_year = st.selectbox(
                "End Year",
                years,
                index=len(years) - 1,
                key="heat_end_year"
            )

        if start_year > end_year:

            st.warning(
                "Start year must be less than or equal to end year."
            )

            data = pd.DataFrame()

        else:

            data = weather[
                (weather["city"] == selected_city) &
                (weather["year"].between(
                    start_year,
                    end_year
                ))
            ].copy()

        period_text = f"{start_year}–{end_year}"

    if not data.empty:

        monthly_stats = (
            data
            .groupby("month_num")
            .agg(
                Avg_Temperature=("tempC", "mean"),
                Avg_Heat_Index=("HeatIndexC", "mean")
            )
            .reset_index()
        )

        monthly_stats[
            "Heat_Stress_Difference"
        ] = (
            monthly_stats["Avg_Heat_Index"]
            - monthly_stats["Avg_Temperature"]
        )

        monthly_stats["Month"] = (
            monthly_stats["month_num"].map(month_map)
        )

        monthly_stats = monthly_stats.sort_values(
            "month_num"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=monthly_stats["Month"],
                y=monthly_stats["Avg_Temperature"],
                mode="lines+markers",
                name="Average Temperature",
                customdata=np.stack(
                    [
                        monthly_stats["Avg_Temperature"],
                        monthly_stats["Avg_Heat_Index"],
                        monthly_stats["Heat_Stress_Difference"]
                    ],
                    axis=-1
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Average Temperature: %{customdata[0]:.2f}°C<br>"
                    "Average Heat Index: %{customdata[1]:.2f}°C<br>"
                    "Heat Stress Difference: %{customdata[2]:.2f}°C"
                    "<extra></extra>"
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=monthly_stats["Month"],
                y=monthly_stats["Avg_Heat_Index"],
                mode="lines+markers",
                name="Average Heat Index"
            )
        )

        fig.update_layout(
            title=(
                f"Monthly Temperature vs Heat Index — "
                f"{selected_city} ({period_text})"
            ),
            xaxis_title="Month",
            yaxis_title="Temperature (°C)",
            hovermode="x unified",
            xaxis=dict(
                categoryorder="array",
                categoryarray=month_names
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Monthly Heat Stress Data")

        display_stats = monthly_stats[
            [
                "Month",
                "Avg_Temperature",
                "Avg_Heat_Index",
                "Heat_Stress_Difference"
            ]
        ].round(2)

        st.dataframe(
            display_stats,
            use_container_width=True,
            hide_index=True
        )

        max_stress = monthly_stats.loc[
            monthly_stats[
                "Heat_Stress_Difference"
            ].idxmax()
        ]

        max_heat_index = monthly_stats.loc[
            monthly_stats[
                "Avg_Heat_Index"
            ].idxmax()
        ]

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Highest Heat Stress Difference",
                f"{max_stress['Heat_Stress_Difference']:.2f} °C",
                max_stress["Month"]
            )

        with col2:

            st.metric(
                "Highest Average Heat Index",
                f"{max_heat_index['Avg_Heat_Index']:.2f} °C",
                max_heat_index["Month"]
            )

    else:

        st.warning(
            "No weather data available for the selected criteria."
        )


    # ========================================================
    # 8. HUMIDITY VS HEAT INDEX
    # ========================================================

    st.header("8. Humidity vs Heat Index Relationship")

    correlation = weather[
        ["humidity", "HeatIndexC"]
    ].corr().iloc[0, 1]

    st.metric(
        "Overall Humidity–Heat Index Correlation",
        f"{correlation:.3f}"
    )

    scatter_data = weather[
        ["humidity", "HeatIndexC", "city", "year"]
    ].dropna()

    fig = px.scatter(
        scatter_data,
        x="humidity",
        y="HeatIndexC",
        color="city",
        opacity=0.35,
        title="Humidity vs Heat Index",
        labels={
            "humidity": "Humidity (%)",
            "HeatIndexC": "Heat Index (°C)",
            "city": "City"
        },
        hover_data={
            "city": True,
            "year": True,
            "humidity": ":.1f",
            "HeatIndexC": ":.2f"
        }
    )

    fig.update_layout(
        hovermode="closest"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # CITY-WISE CORRELATION
    # ========================================================

    st.subheader(
        "City-wise Humidity vs Heat Index Correlation"
    )

    city_correlation = (
        weather
        .groupby("city")
        .apply(
            lambda x:
            x["humidity"].corr(
                x["HeatIndexC"]
            )
        )
        .reset_index(name="correlation")
        .sort_values("correlation")
    )

    fig = px.bar(
        city_correlation,
        x="correlation",
        y="city",
        orientation="h",
        title="Humidity vs Heat Index Correlation by City",
        labels={
            "correlation": "Correlation Coefficient",
            "city": "City"
        },
        text="correlation"
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        hovertemplate=(
            "City: %{y}<br>"
            "Correlation: %{x:.3f}"
            "<extra></extra>"
        )
    )

    fig.add_vline(
        x=0,
        line_width=1
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    strongest_city = city_correlation.loc[
        city_correlation["correlation"].abs().idxmax()
    ]

    st.info(
        f"""
        **Key Insights**

        • Overall Humidity–Heat Index correlation:
        **{correlation:.3f}**

        • The strongest absolute city-wise relationship is observed in
        **{strongest_city['city']}**.

        • The relationship between humidity and perceived heat
        varies across cities.
        """
    )


    # ========================================================
    # PAGE 1 FINAL TAKEAWAY
    # ========================================================

    st.divider()

    st.header("📌 Page 1 — Key Takeaways")

    st.markdown(
        """
        - **Nagpur** has the highest average temperature among the analyzed cities.
        - **Delhi** records the highest extreme temperature.
        - **May** is the hottest month, while **January** is the coolest.
        - **Delhi, Jaipur, Kanpur and Nagpur** show strong seasonal variation.
        - **Bombay** has the highest average Heat Index despite not having the highest average air temperature.
        - **Bengaluru** has the lowest average Heat Index.
        - Heat Index provides additional insight into perceived heat beyond air temperature alone.
        - The relationship between humidity and Heat Index varies substantially by city.
        """
    )


# ============================================================
# PAGE 2
# UV, SUNSHINE & RAINFALL
# ============================================================

elif selected_page == "UV, Sunshine & Rainfall":

    st.title("☀️ UV, Sunshine & Rainfall")

    st.markdown(
        """
        Analysis of UV exposure, sunshine duration, cloud cover and
        precipitation patterns across Indian cities from **2009–2019**.
        """
    )


    # ========================================================
    ## ========================================================
    # 1. UV EXPOSURE BY CITY
    # ========================================================

    st.header("1. UV Exposure by City")

    # --------------------------------------------------------
    # Prepare clean data
    # --------------------------------------------------------

    uv_data = weather.copy()

    # Make sure year is numeric
    uv_data["year"] = pd.to_numeric(
        uv_data["year"],
        errors="coerce"
    )

    # Make sure UV Index is numeric
    uv_data["uvIndex"] = pd.to_numeric(
        uv_data["uvIndex"],
        errors="coerce"
    )

    # Remove invalid rows
    uv_data = uv_data.dropna(
        subset=["city", "year", "uvIndex"]
    )

    # Filter analysis period
    uv_data = uv_data[
        uv_data["year"].between(2009, 2019)
    ].copy()


    # --------------------------------------------------------
    # City-level UV KPIs
    # --------------------------------------------------------

    city_uv = (
        uv_data
        .groupby("city")
        .agg(
            Avg_UV=("uvIndex", "mean"),
            Max_UV=("uvIndex", "max")
        )
        .reset_index()
    )

    # Sort by average UV
    city_uv = city_uv.sort_values(
        "Avg_UV",
        ascending=False
    )


    # --------------------------------------------------------
    # Convert to long format for Plotly
    # --------------------------------------------------------

    city_uv_plot = city_uv.melt(
        id_vars="city",
        value_vars=[
            "Avg_UV",
            "Max_UV"
        ],
        var_name="UV_Measure",
        value_name="UV_Index"
    )

    # Friendly labels
    city_uv_plot["UV_Measure"] = (
        city_uv_plot["UV_Measure"]
        .replace({
            "Avg_UV": "Average UV Index",
            "Max_UV": "Maximum UV Index"
        })
    )


    # --------------------------------------------------------
    # Plotly Bar Chart
    # --------------------------------------------------------

    fig = px.bar(
        city_uv_plot,
        x="city",
        y="UV_Index",
        color="UV_Measure",
        barmode="group",

        title="UV Exposure by City (2009–2019)",

        labels={
            "city": "City",
            "UV_Index": "UV Index",
            "UV_Measure": "UV Measure"
        },

        hover_data={
            "city": True,
            "UV_Measure": True,
            "UV_Index": ":.2f"
        }
    )


    # --------------------------------------------------------
    # Improve Hover Information
    # --------------------------------------------------------

    fig.update_traces(
        hovertemplate=
            "<b>%{x}</b><br>"
            "Measure: %{fullData.name}<br>"
            "UV Index: %{y:.2f}"
            "<extra></extra>"
    )


    # --------------------------------------------------------
    # Chart Layout
    # --------------------------------------------------------

    fig.update_layout(
        height=500,

        xaxis=dict(
            title="City",
            type="category",
            categoryorder="total descending",
            tickangle=-45
        ),

        yaxis=dict(
            title="UV Index",
            rangemode="tozero"
        ),

        hovermode="x unified",

        legend=dict(
            title="UV Measure"
        ),

        margin=dict(
            l=60,
            r=30,
            t=80,
            b=100
        )
    )


    # --------------------------------------------------------
    # Display Chart
    # --------------------------------------------------------

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # UV KPI INSIGHTS
    # ========================================================

    highest_avg_uv_city = city_uv.loc[
        city_uv["Avg_UV"].idxmax()
    ]

    lowest_avg_uv_city = city_uv.loc[
        city_uv["Avg_UV"].idxmin()
    ]

    highest_max_uv = city_uv["Max_UV"].max()

    highest_uv_cities = ", ".join(
        city_uv.loc[
            city_uv["Max_UV"] == highest_max_uv,
            "city"
        ].astype(str)
    )


    # --------------------------------------------------------
    # Insight Box
    # --------------------------------------------------------

    st.info(
        f"""
        **Key Insights**

        • **{highest_avg_uv_city['city']}** has the highest average UV exposure
        at approximately **{highest_avg_uv_city['Avg_UV']:.2f}**.

        • **{lowest_avg_uv_city['city']}** has the lowest average UV exposure
        at approximately **{lowest_avg_uv_city['Avg_UV']:.2f}**.

        • The highest observed UV Index is **{highest_max_uv:.0f}**,
        recorded in **{highest_uv_cities}**.

        • Average exposure and peak exposure can produce different
        city rankings.
        """
    )
    # ========================================================
    # 2. MONTHLY UV EXPOSURE
    # ========================================================

    st.header("2. Monthly UV Exposure")

    monthly_uv = (
        weather[
            weather["year"].between(2009, 2019)
        ]
        .groupby("month_num")
        .agg(
            Avg_UV=("uvIndex", "mean"),
            Max_UV=("uvIndex", "max")
        )
        .reset_index()
    )

    monthly_uv["Month"] = (
        monthly_uv["month_num"].map(month_map)
    )

    monthly_uv = monthly_uv.sort_values(
        "month_num"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly_uv["Month"],
            y=monthly_uv["Avg_UV"],
            mode="lines+markers",
            name="Average UV",
            hovertemplate=(
                "Month: %{x}<br>"
                "Average UV: %{y:.2f}"
                "<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=monthly_uv["Month"],
            y=monthly_uv["Max_UV"],
            mode="lines+markers",
            name="Maximum UV",
            hovertemplate=(
                "Month: %{x}<br>"
                "Maximum UV: %{y:.0f}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title="Monthly UV Exposure (2009–2019)",
        xaxis_title="Month",
        yaxis_title="UV Index",
        hovermode="x unified",
        xaxis=dict(
            categoryorder="array",
            categoryarray=month_names
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    highest_uv_month = monthly_uv.loc[
        monthly_uv["Avg_UV"].idxmax()
    ]

    lowest_uv_month = monthly_uv.loc[
        monthly_uv["Avg_UV"].idxmin()
    ]

    st.info(
        f"""
        **Key Insights**

        • **{highest_uv_month['Month']}** has the highest average UV exposure
        at approximately **{highest_uv_month['Avg_UV']:.2f}**.

        • **{lowest_uv_month['Month']}** has the lowest average UV exposure
        at approximately **{lowest_uv_month['Avg_UV']:.2f}**.

        • UV exposure generally increases during spring and early summer.

        • UV exposure declines after the summer period.
        """
    )


    # ========================================================
    # 3. MONTHLY UV EXPOSURE BY CITY — YEAR-WISE
    # ========================================================

    st.header(
        "3. Monthly UV Exposure by City — Year-wise Analysis"
    )

    monthly_uv_city = (
        weather[
            weather["year"].between(2009, 2019)
        ]
        .groupby(
            ["year", "city", "month_num"]
        )["uvIndex"]
        .mean()
        .reset_index()
    )

    monthly_uv_city["Month"] = (
        monthly_uv_city["month_num"].map(month_map)
    )

    years_uv = sorted(
        monthly_uv_city["year"].dropna().unique().astype(int)
    )

    cities_uv = sorted(
        monthly_uv_city["city"].dropna().unique()
    )

    selected_uv_year = st.selectbox(
        "Select Year",
        years_uv,
        key="uv_year"
    )

    year_uv_data = monthly_uv_city[
        monthly_uv_city["year"] == selected_uv_year
    ].sort_values("month_num")

    fig = px.line(
        year_uv_data,
        x="Month",
        y="uvIndex",
        color="city",
        markers=True,
        title=f"Monthly UV Index by City ({int(selected_uv_year)})",
        labels={
            "uvIndex": "Average UV Index",
            "Month": "Month",
            "city": "City"
        },
        hover_data={
            "uvIndex": ":.2f"
        }
    )

    fig.update_layout(
        hovermode="x unified",
        xaxis=dict(
            categoryorder="array",
            categoryarray=month_names
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    peak_city_year = year_uv_data.loc[
        year_uv_data["uvIndex"].idxmax()
    ]

    st.info(
        f"""
        **Key Insights**

        • In **{int(selected_uv_year)}**, the highest monthly average UV
        value was approximately **{peak_city_year['uvIndex']:.2f}**.

        • This peak occurred in **{peak_city_year['city']}**
        during **{peak_city_year['Month']}**.

        • The chart allows comparison of the seasonal UV pattern
        across all cities for the selected year.
        """
    )


    # ========================================================
    # 4. AVERAGE SUNSHINE DURATION BY CITY
    # ========================================================

    st.header("4. Average Sunshine Duration by City")

    city_sunshine = (
        weather[
            weather["year"].between(2009, 2019)
        ]
        .groupby("city")["sunHour"]
        .mean()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    fig = px.bar(
        city_sunshine,
        x="city",
        y="sunHour",
        title="Average Sunshine Duration by City",
        labels={
            "city": "City",
            "sunHour": "Average Sunshine Hours"
        },
        text="sunHour"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        hovertemplate=(
            "City: %{x}<br>"
            "Average Sunshine: %{y:.2f} hours/day"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    highest_sun_city = city_sunshine.loc[
        city_sunshine["sunHour"].idxmax()
    ]

    lowest_sun_city = city_sunshine.loc[
        city_sunshine["sunHour"].idxmin()
    ]

    st.info(
        f"""
        **Key Insights**

        • **{highest_sun_city['city']}** has the highest average sunshine
        duration at approximately **{highest_sun_city['sunHour']:.2f} hours/day**.

        • **{lowest_sun_city['city']}** records the lowest average sunshine
        duration at approximately **{lowest_sun_city['sunHour']:.2f} hours/day**.

        • The difference between the highest and lowest city is approximately
        **{highest_sun_city['sunHour'] - lowest_sun_city['sunHour']:.2f} hours/day**.
        """
    )


    # ========================================================
    # 5. CLOUD COVER VS SUNSHINE HOURS
    # ========================================================

    st.header("5. Cloud Cover vs Sunshine Hours")

    cloud_sun_data = weather[
        [
            "cloudcover",
            "sunHour",
            "city",
            "year"
        ]
    ].dropna()

    cloud_sun_corr = cloud_sun_data[
        ["cloudcover", "sunHour"]
    ].corr().iloc[0, 1]

    fig = px.scatter(
        cloud_sun_data,
        x="cloudcover",
        y="sunHour",
        color="city",
        opacity=0.35,
        title="Cloud Cover vs Sunshine Hours",
        labels={
            "cloudcover": "Cloud Cover (%)",
            "sunHour": "Sunshine Hours",
            "city": "City"
        },
        hover_data={
            "city": True,
            "year": True,
            "cloudcover": ":.1f",
            "sunHour": ":.2f"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.metric(
        "Cloud Cover – Sunshine Correlation",
        f"{cloud_sun_corr:.3f}"
    )

    st.info(
        f"""
        **Key Insight**

        • Cloud cover and sunshine duration have a correlation of
        approximately **{cloud_sun_corr:.3f}**.

        • The negative relationship indicates that higher cloud cover
        generally corresponds to lower sunshine duration.

        • Cloud cover alone does not explain all variation in sunshine hours.
        """
    )


    # ========================================================
    # 6. TOTAL PRECIPITATION BY CITY
    # ========================================================

    st.header("6. Total Precipitation by City (2009–2019)")

    city_rainfall = (
        weather[
            weather["year"].between(2009, 2019)
        ]
        .groupby("city")["precipMM"]
        .sum()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    fig = px.bar(
        city_rainfall,
        x="city",
        y="precipMM",
        title="Total Precipitation by City (2009–2019)",
        labels={
            "city": "City",
            "precipMM": "Total Precipitation (mm)"
        },
        text="precipMM"
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        hovertemplate=(
            "City: %{x}<br>"
            "Total Precipitation: %{y:,.1f} mm"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    highest_rain_city = city_rainfall.loc[
        city_rainfall["precipMM"].idxmax()
    ]

    lowest_rain_city = city_rainfall.loc[
        city_rainfall["precipMM"].idxmin()
    ]

    st.info(
        f"""
        **Key Insights**

        • **{highest_rain_city['city']}** records the highest cumulative
        precipitation at approximately **{highest_rain_city['precipMM']:,.1f} mm**.

        • **{lowest_rain_city['city']}** records the lowest cumulative
        precipitation at approximately **{lowest_rain_city['precipMM']:,.1f} mm**.

        • There are substantial differences in cumulative precipitation
        across the eight cities.
        """
    )


    # ========================================================
    # 7. MONTHLY PRECIPITATION
    # ========================================================

    st.header("7. Monthly Precipitation")

    monthly_rainfall = (
        weather[
            weather["year"].between(2009, 2019)
        ]
        .groupby("month_num")["precipMM"]
        .sum()
        .reindex(range(1, 13))
        .reset_index()
    )

    monthly_rainfall["Month"] = (
        monthly_rainfall["month_num"].map(month_map)
    )

    fig = px.bar(
        monthly_rainfall,
        x="Month",
        y="precipMM",
        title="Monthly Precipitation (2009–2019)",
        labels={
            "Month": "Month",
            "precipMM": "Total Precipitation (mm)"
        },
        text="precipMM"
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        hovertemplate=(
            "Month: %{x}<br>"
            "Total Precipitation: %{y:,.1f} mm"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        xaxis=dict(
            categoryorder="array",
            categoryarray=month_names
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    wettest_month = monthly_rainfall.loc[
        monthly_rainfall["precipMM"].idxmax()
    ]

    driest_month = monthly_rainfall.loc[
        monthly_rainfall["precipMM"].idxmin()
    ]

    st.info(
        f"""
        **Key Insights**

        • **{wettest_month['Month']}** records the highest cumulative
        precipitation at approximately **{wettest_month['precipMM']:,.1f} mm**.

        • **{driest_month['Month']}** records the lowest cumulative
        precipitation at approximately **{driest_month['precipMM']:,.1f} mm**.

        • Rainfall is strongly concentrated during the monsoon period.
        """
    )


    # ========================================================
    # 8. HIGHEST DAILY RAINFALL
    # ========================================================

    st.header("8. Highest Daily Rainfall Event by City")

    daily_rainfall = (
        weather[
            weather["year"].between(2009, 2019)
        ]
        .groupby(
            ["city", "date"]
        )["precipMM"]
        .sum()
        .reset_index(
            name="daily_precipMM"
        )
    )

    max_daily_rainfall = (
        daily_rainfall
        .groupby("city")["daily_precipMM"]
        .max()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    fig = px.bar(
        max_daily_rainfall,
        x="city",
        y="daily_precipMM",
        title="Highest Daily Rainfall Event by City",
        labels={
            "city": "City",
            "daily_precipMM": "Maximum Daily Rainfall (mm)"
        },
        text="daily_precipMM"
    )

    fig.update_traces(
        texttemplate="%{text:.1f} mm",
        textposition="outside",
        hovertemplate=(
            "City: %{x}<br>"
            "Maximum Daily Rainfall: %{y:.1f} mm"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    highest_daily_city = max_daily_rainfall.loc[
        max_daily_rainfall["daily_precipMM"].idxmax()
    ]

    lowest_daily_city = max_daily_rainfall.loc[
        max_daily_rainfall["daily_precipMM"].idxmin()
    ]

    st.info(
        f"""
        **Key Insights**

        • **{highest_daily_city['city']}** experienced the highest
        single-day rainfall event at approximately
        **{highest_daily_city['daily_precipMM']:.1f} mm**.

        • **{lowest_daily_city['city']}** recorded the lowest maximum
        daily rainfall at approximately
        **{lowest_daily_city['daily_precipMM']:.1f} mm**.

        • The results highlight differences in extreme rainfall
        intensity across cities.
        """
    )


    # ========================================================
    # PAGE 2 FINAL TAKEAWAY
    # ========================================================

    st.divider()

    st.header("📌 Page 2 — Key Takeaways")

    st.markdown(
        """
        - **Kanpur** has the highest average UV exposure, while
          **Bengaluru** has the lowest.
        - UV exposure follows a strong seasonal pattern, generally
          peaking during **April–June**.
        - **Jaipur and Delhi** have high average sunshine duration.
        - Cloud cover has a negative relationship with sunshine duration.
        - **Bombay** records the highest cumulative precipitation.
        - **Jaipur** records the lowest cumulative precipitation.
        - Rainfall is concentrated mainly during the **monsoon period**.
        - **Bombay** experienced one of the strongest single-day rainfall events.
        """
    )


# ============================================================
# PAGE 3
# WIND, VISIBILITY & RELATIONSHIPS
# ============================================================

elif page == "Wind, Visibility & Relationships":

    st.title(
        "💨 Wind, Visibility & Relationships"
    )

    st.markdown(
        """
        Analysis of wind speed, wind gusts, prevailing wind direction,
        visibility trends and relationships between weather variables
        across Indian cities from **2009–2019**.
        """
    )


    # ========================================================
    # # ========================================================
    # ========================================================
    # ========================================================
    # 1. MAXIMUM WIND SPEED & GUST BY CITY
    # ========================================================

    st.header("1. Maximum Wind Speed and Gust by City")

    # --------------------------------------------------------
    # Calculate exactly as in your Python analysis
    # --------------------------------------------------------

    city_wind = (
        weather
        .groupby("city")
        .agg(
            Max_WindSpeed=("windspeedKmph", "max"),
            Max_WindGust=("WindGustKmph", "max")
        )
        .sort_values(
            "Max_WindSpeed",
            ascending=False
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Convert to long format for Plotly
    # --------------------------------------------------------

    wind_plot = city_wind.melt(
        id_vars="city",
        value_vars=[
            "Max_WindSpeed",
            "Max_WindGust"
        ],
        var_name="Wind Measure",
        value_name="Wind Speed (km/h)"
    )

    # Rename labels
    wind_plot["Wind Measure"] = wind_plot[
        "Wind Measure"
    ].replace({
        "Max_WindSpeed": "Maximum Wind Speed",
        "Max_WindGust": "Maximum Wind Gust"
    })

    # --------------------------------------------------------
    # Plotly chart
    # --------------------------------------------------------

    fig = px.bar(
        wind_plot,
        x="city",
        y="Wind Speed (km/h)",
        color="Wind Measure",
        barmode="group",

        title="Maximum Wind Speed and Gust by City (2009–2019)",

        labels={
            "city": "City",
            "Wind Speed (km/h)": "Wind Speed (km/h)",
            "Wind Measure": "Wind Measure"
        },

        hover_data={
            "city": True,
            "Wind Measure": True,
            "Wind Speed (km/h)": ":.0f"
        }
    )

    # --------------------------------------------------------
    # Chart formatting
    # --------------------------------------------------------

    fig.update_layout(
        height=550,

        xaxis=dict(
            title="City",
            categoryorder="array",
            categoryarray=city_wind["city"].tolist(),
            tickangle=-45
        ),

        yaxis=dict(
            title="Wind Speed (km/h)",
            rangemode="tozero"
        ),

        hovermode="x unified",

        legend=dict(
            title="Wind Measure"
        )
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # KEY INSIGHTS
    # ========================================================

    highest_wind_speed_city = city_wind.loc[
        city_wind["Max_WindSpeed"].idxmax()
    ]

    highest_wind_gust_city = city_wind.loc[
        city_wind["Max_WindGust"].idxmax()
    ]

    lowest_wind_speed_city = city_wind.loc[
        city_wind["Max_WindSpeed"].idxmin()
    ]

    st.info(
        f"""
        **Key Insights**

        • **{highest_wind_gust_city['city']}** recorded the highest
        maximum wind gust at **{highest_wind_gust_city['Max_WindGust']:.0f} km/h**.

        • **{highest_wind_speed_city['city']}** recorded the highest
        maximum wind speed at **{highest_wind_speed_city['Max_WindSpeed']:.0f} km/h**.

        • **{lowest_wind_speed_city['city']}** recorded the lowest
        maximum wind speed at **{lowest_wind_speed_city['Max_WindSpeed']:.0f} km/h**.

        • Maximum wind speed and maximum wind gust are different KPIs,
        so both are useful for identifying extreme wind conditions.
        """
    )
    # ========================================================
    # 2. MONTHLY WIND SPEED & GUST
    # ========================================================

    st.header("2. Monthly Wind Speed & Gust")

    st.markdown(
        "Select a city and analysis period."
    )

    wind_cities = sorted(
        weather["city"].dropna().unique()
    )

    wind_years = sorted(
        weather["year"].dropna().astype(int).unique()
    )

    selected_wind_city = st.selectbox(
        "Select City",
        wind_cities,
        key="wind_city"
    )

    wind_analysis_type = st.radio(
        "Analysis Period",
        [
            "Specific Year",
            "Year Range"
        ],
        horizontal=True,
        key="wind_period"
    )

    if wind_analysis_type == "Specific Year":

        selected_wind_year = st.selectbox(
            "Select Year",
            wind_years,
            key="wind_year"
        )

        filtered_weather = weather[
            (weather["city"] == selected_wind_city) &
            (weather["year"] == selected_wind_year)
        ].copy()

        wind_period_text = str(
            selected_wind_year
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            wind_start_year = st.selectbox(
                "Start Year",
                wind_years,
                index=0,
                key="wind_start_year"
            )

        with col2:

            wind_end_year = st.selectbox(
                "End Year",
                wind_years,
                index=len(wind_years) - 1,
                key="wind_end_year"
            )

        if wind_start_year > wind_end_year:

            st.warning(
                "Start year must be less than or equal to end year."
            )

            filtered_weather = pd.DataFrame()

        else:

            filtered_weather = weather[
                (weather["city"] == selected_wind_city) &
                (
                    weather["year"].between(
                        wind_start_year,
                        wind_end_year
                    )
                )
            ].copy()

        wind_period_text = (
            f"{wind_start_year}–{wind_end_year}"
        )


    if not filtered_weather.empty:

        monthly_wind = (
            filtered_weather
            .groupby("month_num")
            .agg(
                Avg_WindSpeed=("windspeedKmph", "mean"),
                Max_WindSpeed=("windspeedKmph", "max"),
                Avg_WindGust=("WindGustKmph", "mean"),
                Max_WindGust=("WindGustKmph", "max")
            )
            .reset_index()
        )

        monthly_wind["Month"] = (
            monthly_wind["month_num"].map(month_map)
        )

        monthly_wind = monthly_wind.sort_values(
            "month_num"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=monthly_wind["Month"],
                y=monthly_wind["Avg_WindSpeed"],
                mode="lines+markers",
                name="Average Wind Speed",
                customdata=np.stack(
                    [
                        monthly_wind["Max_WindSpeed"],
                        monthly_wind["Avg_WindGust"],
                        monthly_wind["Max_WindGust"]
                    ],
                    axis=-1
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Average Wind Speed: %{y:.2f} km/h<br>"
                    "Maximum Wind Speed: %{customdata[0]:.1f} km/h<br>"
                    "Average Wind Gust: %{customdata[1]:.2f} km/h<br>"
                    "Maximum Wind Gust: %{customdata[2]:.1f} km/h"
                    "<extra></extra>"
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=monthly_wind["Month"],
                y=monthly_wind["Avg_WindGust"],
                mode="lines+markers",
                name="Average Wind Gust"
            )
        )

        fig.update_layout(
            title=(
                f"Monthly Wind Speed & Gust — "
                f"{selected_wind_city} ({wind_period_text})"
            ),
            xaxis_title="Month",
            yaxis_title="Wind Speed (km/h)",
            hovermode="x unified",
            xaxis=dict(
                categoryorder="array",
                categoryarray=month_names
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Monthly Wind Statistics")

        display_wind = monthly_wind[
            [
                "Month",
                "Avg_WindSpeed",
                "Max_WindSpeed",
                "Avg_WindGust",
                "Max_WindGust"
            ]
        ].round(2)

        st.dataframe(
            display_wind,
            use_container_width=True,
            hide_index=True
        )

        highest_avg_wind = monthly_wind.loc[
            monthly_wind["Avg_WindSpeed"].idxmax()
        ]

        highest_max_wind = monthly_wind.loc[
            monthly_wind["Max_WindSpeed"].idxmax()
        ]

        highest_max_gust = monthly_wind.loc[
            monthly_wind["Max_WindGust"].idxmax()
        ]

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Highest Avg Wind Speed",
                f"{highest_avg_wind['Avg_WindSpeed']:.2f} km/h",
                highest_avg_wind["Month"]
            )

        with col2:

            st.metric(
                "Highest Max Wind Speed",
                f"{highest_max_wind['Max_WindSpeed']:.0f} km/h",
                highest_max_wind["Month"]
            )

        with col3:

            st.metric(
                "Highest Max Wind Gust",
                f"{highest_max_gust['Max_WindGust']:.0f} km/h",
                highest_max_gust["Month"]
            )

    else:

        st.warning(
            "No weather data available for the selected criteria."
        )


    # ========================================================
    # 3. DOMINANT WIND DIRECTION BY CITY
    # ========================================================

    st.header("3. Dominant Wind Direction by City")

    def get_wind_direction(degree):

        if pd.isna(degree):
            return np.nan

        if degree >= 337.5 or degree < 22.5:
            return "N"

        elif degree < 67.5:
            return "NE"

        elif degree < 112.5:
            return "E"

        elif degree < 157.5:
            return "SE"

        elif degree < 202.5:
            return "S"

        elif degree < 247.5:
            return "SW"

        elif degree < 292.5:
            return "W"

        else:
            return "NW"


    weather["wind_direction"] = (
        weather["winddirDegree"]
        .apply(get_wind_direction)
    )

    direction_counts = (
        weather
        .dropna(subset=["wind_direction"])
        .groupby(
            ["city", "wind_direction"]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    dominant_direction = (
        direction_counts
        .loc[
            direction_counts
            .groupby("city")["count"]
            .idxmax()
        ]
        .reset_index(drop=True)
    )

    dominant_direction = dominant_direction.sort_values(
        "count",
        ascending=False
    )

    fig = px.bar(
        dominant_direction,
        x="city",
        y="count",
        color="wind_direction",
        title="Dominant Wind Direction by City",
        labels={
            "city": "City",
            "count": "Number of Observations",
            "wind_direction": "Dominant Direction"
        },
        text="wind_direction"
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "City: %{x}<br>"
            "Dominant Direction: %{text}<br>"
            "Observations: %{y:,}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    dominant_table = dominant_direction[
        [
            "city",
            "wind_direction",
            "count"
        ]
    ].rename(
        columns={
            "city": "City",
            "wind_direction": "Dominant Direction",
            "count": "Observation Count"
        }
    )

    st.dataframe(
        dominant_table,
        use_container_width=True,
        hide_index=True
    )

    west_count = (
        dominant_direction[
            dominant_direction["wind_direction"] == "W"
        ].shape[0]
    )

    st.info(
        f"""
        **Key Insights**

        • Westerly (**W**) winds are the dominant direction in
        **{west_count} of the analyzed cities**.

        • **Bombay** shows a dominant south-westerly (**SW**) pattern.

        • **Delhi and Kanpur** show dominant north-westerly (**NW**) winds.

        • These counts represent the frequency of observations,
        not wind strength.
        """
    )


    # ========================================================
    # # ========================================================
    # 4. AVERAGE VISIBILITY BY CITY
    # ========================================================

    st.header("4. Average Visibility by City")


    # --------------------------------------------------------
    # Filter analysis period
    # --------------------------------------------------------

    visibility_data = weather[
        weather["year"].between(2009, 2019)
    ].copy()


    # --------------------------------------------------------
    # Make visibility numeric
    # --------------------------------------------------------

    visibility_data["visibility"] = pd.to_numeric(
        visibility_data["visibility"],
        errors="coerce"
    )


    # Remove missing values
    visibility_data = visibility_data.dropna(
        subset=[
            "city",
            "visibility"
        ]
    )


    # --------------------------------------------------------
    # City-level visibility KPIs
    # --------------------------------------------------------

    city_visibility = (
        visibility_data
        .groupby("city", as_index=False)
        .agg(
            Avg_Visibility=("visibility", "mean"),
            Min_Visibility=("visibility", "min")
        )
    )


    # Sort by average visibility
    city_visibility = city_visibility.sort_values(
        "Avg_Visibility",
        ascending=True
    )


    # --------------------------------------------------------
    # Convert to long format for Plotly
    # --------------------------------------------------------

    visibility_long = city_visibility.melt(
        id_vars="city",
        value_vars=[
            "Avg_Visibility",
            "Min_Visibility"
        ],
        var_name="Visibility Measure",
        value_name="Visibility"
    )


    # Rename measures
    visibility_long["Visibility Measure"] = (
        visibility_long["Visibility Measure"]
        .replace({
            "Avg_Visibility": "Average Visibility",
            "Min_Visibility": "Minimum Visibility"
        })
    )


    # --------------------------------------------------------
    # Plotly bar chart
    # --------------------------------------------------------

    fig = px.bar(
        visibility_long,
        x="city",
        y="Visibility",
        color="Visibility Measure",
        barmode="group",

        title="Average and Minimum Visibility by City (2009–2019)",

        labels={
            "city": "City",
            "Visibility": "Visibility (km)",
            "Visibility Measure": "Visibility Measure"
        },

        hover_data={
            "city": True,
            "Visibility Measure": True,
            "Visibility": ":.2f"
        }
    )


    # --------------------------------------------------------
    # Custom hover
    # --------------------------------------------------------

    fig.update_traces(
        hovertemplate=
        "<b>City:</b> %{x}<br>"
        "<b>%{fullData.name}:</b> %{y:.2f} km"
        "<extra></extra>"
    )


    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_layout(
        height=550,

        xaxis=dict(
            title="City",
            type="category",
            categoryorder="total ascending",
            tickangle=-45
        ),

        yaxis=dict(
            title="Visibility (km)",
            rangemode="tozero"
        ),

        hovermode="x",

        legend=dict(
            title="Visibility Measure"
        ),

        margin=dict(
            l=60,
            r=30,
            t=80,
            b=100
        )
    )


    # --------------------------------------------------------
    # Display chart
    # --------------------------------------------------------

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # KEY INSIGHTS
    # ========================================================

    lowest_visibility_city = city_visibility.loc[
        city_visibility["Avg_Visibility"].idxmin()
    ]

    highest_visibility_city = city_visibility.loc[
        city_visibility["Avg_Visibility"].idxmax()
    ]


    # Calculate difference
    visibility_difference = (
        highest_visibility_city["Avg_Visibility"]
        - lowest_visibility_city["Avg_Visibility"]
    )


    # --------------------------------------------------------
    # Insight box
    # --------------------------------------------------------

    st.info(
        f"""
        **Key Insights**

        • **{lowest_visibility_city['city']}** has the lowest average
        visibility at approximately **{lowest_visibility_city['Avg_Visibility']:.2f} km**.

        • **{highest_visibility_city['city']}** has the highest average
        visibility at approximately **{highest_visibility_city['Avg_Visibility']:.2f} km**.

        • The difference between the highest and lowest city averages
        is approximately **{visibility_difference:.2f} km**.

        • Several cities recorded extremely low individual visibility events,
        but minimum visibility represents extreme observations rather than
        typical city conditions.
        """
    )

    # ========================================================
    # 5. ANNUAL VISIBILITY TREND
    # ========================================================

    st.header("5. Annual Visibility Trend by City")

    visibility_cities = sorted(
        weather["city"].dropna().unique()
    )

    selected_visibility_city = st.selectbox(
        "Select City",
        visibility_cities,
        key="visibility_city"
    )

    city_visibility_yearly = (
        weather
        .groupby(
            ["city", "year"]
        )
        .agg(
            Avg_Visibility=("visibility", "mean"),
            Min_Visibility=("visibility", "min")
        )
        .reset_index()
    )

    visibility_data = (
        city_visibility_yearly[
            city_visibility_yearly["city"]
            == selected_visibility_city
        ]
        .sort_values("year")
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=visibility_data["year"],
            y=visibility_data["Avg_Visibility"],
            mode="lines+markers",
            name="Average Visibility",
            hovertemplate=(
                "Year: %{x}<br>"
                "Average Visibility: %{y:.2f} km"
                "<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=visibility_data["year"],
            y=visibility_data["Min_Visibility"],
            mode="lines+markers",
            name="Minimum Visibility",
            hovertemplate=(
                "Year: %{x}<br>"
                "Minimum Visibility: %{y:.2f} km"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=(
            f"Annual Visibility Trend — "
            f"{selected_visibility_city}"
        ),
        xaxis_title="Year",
        yaxis_title="Visibility (km)",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Annual Visibility Data")

    st.dataframe(
        visibility_data[
            [
                "year",
                "Avg_Visibility",
                "Min_Visibility"
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True
    )

    first_year_visibility = visibility_data.iloc[0]
    last_year_visibility = visibility_data.iloc[-1]

    visibility_change = (
        last_year_visibility["Avg_Visibility"]
        - first_year_visibility["Avg_Visibility"]
    )

    if visibility_change > 0:

        trend_text = "improved"

    elif visibility_change < 0:

        trend_text = "declined"

    else:

        trend_text = "remained stable"

    st.info(
        f"""
        **Key Insight**

        • Average visibility in **{selected_visibility_city}**
        {trend_text} from approximately
        **{first_year_visibility['Avg_Visibility']:.2f} km**
        in **{int(first_year_visibility['year'])}** to
        **{last_year_visibility['Avg_Visibility']:.2f} km**
        in **{int(last_year_visibility['year'])}**.

        • Change in average visibility:
        **{visibility_change:+.2f} km**.
        """
    )


    # ========================================================
    # 6. HUMIDITY VS VISIBILITY
    # ========================================================

    st.header("6. Humidity vs Visibility Relationship")

    humidity_visibility_data = weather[
        [
            "humidity",
            "visibility",
            "city",
            "year"
        ]
    ].dropna()

    humidity_visibility_corr = (
        humidity_visibility_data[
            [
                "humidity",
                "visibility"
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    st.metric(
        "Humidity–Visibility Correlation",
        f"{humidity_visibility_corr:.3f}"
    )

    fig = px.scatter(
        humidity_visibility_data,
        x="humidity",
        y="visibility",
        color="city",
        opacity=0.35,
        title="Humidity vs Visibility",
        labels={
            "humidity": "Humidity (%)",
            "visibility": "Visibility (km)",
            "city": "City"
        },
        hover_data={
            "city": True,
            "year": True,
            "humidity": ":.1f",
            "visibility": ":.2f"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        f"""
        **Key Insights**

        • Humidity–Visibility correlation is
        **{humidity_visibility_corr:.3f}**.

        • The negative relationship indicates that higher humidity
        tends to be associated with lower visibility.

        • Humidity alone does not explain all changes in visibility.

        • Other factors such as precipitation, cloud cover, fog and
        atmospheric conditions can also influence visibility.
        """
    )


    # ========================================================
    # 7. WEATHER VARIABLES CORRELATION MATRIX
    # ========================================================

    st.header("7. Weather Variables Correlation Matrix")

    corr_columns = [
        "maxtempC",
        "mintempC",
        "sunHour",
        "uvIndex",
        "DewPointC",
        "FeelsLikeC",
        "HeatIndexC",
        "WindGustKmph",
        "cloudcover",
        "humidity",
        "precipMM",
        "pressure",
        "tempC",
        "visibility",
        "windspeedKmph"
    ]

    available_corr_columns = [
        col
        for col in corr_columns
        if col in weather.columns
    ]

    correlation_matrix = (
        weather[
            available_corr_columns
        ]
        .corr()
    )

    fig = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        title="Weather Variables Correlation Matrix",
        labels={
            "color": "Correlation"
        }
    )

    fig.update_traces(
        hovertemplate=(
            "Variable 1: %{y}<br>"
            "Variable 2: %{x}<br>"
            "Correlation: %{z:.3f}"
            "<extra></extra>"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # STRONGEST CORRELATIONS
    # ========================================================

    correlation_pairs = []

    for i in range(
        len(correlation_matrix.columns)
    ):

        for j in range(
            i + 1,
            len(correlation_matrix.columns)
        ):

            var1 = correlation_matrix.columns[i]

            var2 = correlation_matrix.columns[j]

            value = correlation_matrix.iloc[
                i,
                j
            ]

            if not pd.isna(value):

                correlation_pairs.append(
                    {
                        "Variable 1": var1,
                        "Variable 2": var2,
                        "Correlation": value,
                        "Absolute Correlation": abs(value)
                    }
                )


    correlation_pairs = pd.DataFrame(
        correlation_pairs
    )

    strongest_pairs = (
        correlation_pairs
        .sort_values(
            "Absolute Correlation",
            ascending=False
        )
        .head(8)
    )

    st.subheader(
        "Strongest Weather Variable Relationships"
    )

    st.dataframe(
        strongest_pairs[
            [
                "Variable 1",
                "Variable 2",
                "Correlation"
            ]
        ].round(3),
        use_container_width=True,
        hide_index=True
    )


    st.info(
        """
        **Important Relationships**

        • **Temperature ↔ Heat Index** shows a very strong positive relationship.

        • **Feels Like ↔ Heat Index** shows an extremely strong positive relationship.

        • **Wind Speed ↔ Wind Gust** shows a very strong positive relationship.

        • **Maximum Temperature ↔ UV Index** shows a very strong positive relationship.

        • **Humidity ↔ Dew Point** shows a strong positive relationship.

        • **Humidity ↔ Visibility** shows a moderate negative relationship.

        • **Cloud Cover ↔ Visibility** shows a moderate negative relationship.

        • **Pressure ↔ Minimum Temperature** shows a strong negative relationship.
        """
    )


    # ========================================================
    # PAGE 3 FINAL TAKEAWAY
    # ========================================================

    st.divider()

    st.header(
        "📌 Page 3 — Key Takeaways"
    )

    st.markdown(
        """
        - **Bombay** recorded the strongest maximum wind gust among the cities.
        - **Bombay and Kanpur** recorded the highest maximum wind speed.
        - **Hyderabad** shows strong sustained wind conditions during the monsoon period.
        - Westerly (**W**) winds dominate most of the analyzed cities.
        - **Bombay** is dominated by south-westerly winds.
        - **Delhi and Kanpur** show dominant north-westerly winds.
        - Average visibility differences between cities are relatively small.
        - Humidity has a negative relationship with visibility.
        - Weather variables such as Temperature, Heat Index, Wind Speed and Wind Gust show strong relationships.
        - Correlation analysis helps identify relationships between weather variables but does not by itself establish causation.
        """
    )

# ============================================================
# PAGE 4 — TEMPERATURE PREDICTION
# ============================================================

elif page == "Temperature Prediction":

    st.title("🌡️ Temperature Prediction")

    st.markdown(
        """
        Enter the weather conditions below to predict temperature
        using the trained Random Forest Regression model.
        """
    )

    st.divider()

    # ========================================================
    # INPUT SECTION
    # ========================================================

    st.subheader("🌤️ Weather Conditions")

    # --------------------------------------------------------
    # CITY
    # --------------------------------------------------------

    city = st.selectbox(
        "City",
        [
            "Bombay",
            "Delhi",
            "Bengaluru",
            "Hyderabad",
            "Pune",
            "Jaipur",
            "Kanpur",
            "Nagpur"
        ]
    )

    # --------------------------------------------------------
    # MONTH
    # --------------------------------------------------------

    month_options = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12
    }

    month_name = st.selectbox(
        "Month",
        list(month_options.keys())
    )

    month_num = month_options[month_name]

    # --------------------------------------------------------
    # SEASON
    # --------------------------------------------------------

    season = st.selectbox(
        "Season",
        [
            "Winter",
            "Spring",
            "Summer",
            "Monsoon"
        ]
    )

    # --------------------------------------------------------
    # HOUR
    # --------------------------------------------------------

    hour = st.slider(
        "Hour",
        min_value=0,
        max_value=23,
        value=12,
        step=1
    )

    # ========================================================
    # WEATHER INPUTS
    # ========================================================

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with col1:

        sunHour = st.number_input(
            "Sunshine Hours",
            min_value=0.0,
            max_value=24.0,
            value=12.0,
            step=0.1
        )

        uvIndex = st.number_input(
            "UV Index",
            min_value=0.0,
            max_value=15.0,
            value=5.0,
            step=0.1
        )

        DewPointC = st.number_input(
            "Dew Point (°C)",
            value=21.0,
            step=0.1
        )

        WindGustKmph = st.number_input(
            "Wind Gust (km/h)",
            min_value=0.0,
            value=20.0,
            step=1.0
        )

        cloudcover = st.number_input(
            "Cloud Cover (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0
        )

    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

    with col2:

        precipMM = st.number_input(
            "Precipitation (mm)",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

        pressure = st.number_input(
            "Pressure (hPa)",
            min_value=800.0,
            max_value=1200.0,
            value=1010.0,
            step=1.0
        )

        visibility = st.number_input(
            "Visibility (km)",
            min_value=0.0,
            value=10.0,
            step=0.1
        )

        winddirDegree = st.number_input(
            "Wind Direction (°)",
            min_value=0.0,
            max_value=360.0,
            value=270.0,
            step=1.0
        )

        windspeedKmph = st.number_input(
            "Wind Speed (km/h)",
            min_value=0.0,
            value=15.0,
            step=1.0
        )

    st.divider()

    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    predict_button = st.button(
        "🌡️ Predict Temperature",
        use_container_width=True
    )

    if predict_button:

        # ====================================================
        # CREATE INPUT DATAFRAME
        # ====================================================

        input_data = pd.DataFrame({

            "hour": [hour],

            "month_num": [month_num],

            "season": [season],

            "city": [city],

            "sunHour": [sunHour],

            "uvIndex": [uvIndex],

            "DewPointC": [DewPointC],

            "WindGustKmph": [WindGustKmph],

            "cloudcover": [cloudcover],

            "humidity": [humidity],

            "precipMM": [precipMM],

            "pressure": [pressure],

            "visibility": [visibility],

            "winddirDegree": [winddirDegree],

            "windspeedKmph": [windspeedKmph]

        })

        # ====================================================
        # PREDICTION
        # ====================================================

        predicted_temperature = rf_model.predict(
            input_data
        )[0]

        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.divider()

        st.subheader("🌡️ Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Predicted Temperature",
                f"{predicted_temperature:.2f} °C"
            )

        with result_col2:

            st.metric(
                "City",
                city
            )

        with result_col3:

            st.metric(
                "Hour",
                f"{hour:02d}:00"
            )

        st.success(
            f"Predicted temperature for {city} at {hour:02d}:00 "
            f"in {month_name} is **{predicted_temperature:.2f} °C**."
        )

        # ====================================================
        # SHOW INPUT SUMMARY
        # ====================================================

        with st.expander("View Input Parameters"):

            display_input = input_data.copy()

            display_input["month"] = month_name

            st.dataframe(
                display_input,
                use_container_width=True
            )
