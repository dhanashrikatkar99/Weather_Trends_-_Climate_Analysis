# 🌦️ Weather Trends & Climate Analysis

An interactive weather data analysis and visualization project built using **Python, Pandas, Plotly, and Streamlit**. The project analyzes historical weather data from **2009–2019 across 8 Indian cities** to identify temperature, heat, UV, wind, visibility, and weather-variable relationships.

---

## 📌 Project Overview

This project performs Exploratory Data Analysis (EDA) on historical weather data and presents the findings through an interactive Streamlit dashboard.

The analysis focuses on:

- 🌡️ Temperature trends and extremes
- ☀️ UV exposure
- 🔥 Heat Index and heat stress
- 💨 Wind speed and wind gusts
- 🧭 Dominant wind direction
- 👁️ Visibility trends
- 💧 Humidity relationships
- 📊 Weather-variable correlations
- 📅 Monthly and seasonal patterns

---

## 🏙️ Cities Analyzed

The dataset contains weather observations for:

- Bengaluru
- Bombay
- Delhi
- Hyderabad
- Jaipur
- Kanpur
- Nagpur
- Pune

**Analysis Period:** 2009–2019

---

## 📊 Dashboard

### Page 1 — Temperature & Heat Analysis

- Temperature Comparison by City
- Highest Recorded Temperature by City
- Monthly Temperature Trend
- Average Temperature by City and Month
- Average Temperature by Season
- Average Heat Index by City
- Monthly Temperature vs Heat Index
- Humidity vs Heat Index Relationship

### Page 2 — UV & Solar Analysis

- UV Exposure by City
- Average and Maximum UV Index comparison
- Monthly UV trends
- City-wise UV analysis
- UV and temperature relationships

### Page 3 — Wind, Visibility & Relationships

- Maximum Wind Speed and Gust by City
- Monthly Wind Speed & Gust Analysis
- Dominant Wind Direction by City
- Average Visibility by City
- Annual Visibility Trend
- Humidity vs Visibility Relationship
- Weather Variables Correlation Matrix

---

## 🔍 Key Analysis

The project uses aggregation, trend analysis, comparative analysis, and correlation analysis to identify meaningful weather patterns.

### Temperature

- Compared minimum, average, and maximum temperatures across cities.
- Identified extreme temperature events.
- Analyzed monthly and seasonal temperature patterns.

### Heat & UV

- Compared average and maximum UV exposure between cities.
- Analyzed Heat Index alongside actual temperature.
- Identified periods with higher potential heat stress.

### Wind

- Compared maximum wind speed and wind gusts.
- Analyzed monthly wind patterns.
- Identified dominant wind directions for each city.

### Visibility

- Compared average and minimum visibility across cities.
- Analyzed annual visibility trends from 2009–2019.
- Studied the relationship between humidity and visibility.

### Correlation Analysis

The project analyzes relationships between variables such as:

- Temperature ↔ Heat Index
- Wind Speed ↔ Wind Gust
- Temperature ↔ UV Index
- Humidity ↔ Visibility
- Cloud Cover ↔ Visibility
- Humidity ↔ Dew Point

---

## 📈 Key Findings

- **Nagpur** recorded the highest average temperature among the analyzed cities.
- **Delhi** recorded the highest extreme temperature at approximately **52°C**.
- **May** was the hottest month, while **January** was the coolest.
- **Bombay** recorded the highest maximum wind gust at approximately **85 km/h**.
- **Hyderabad** recorded the highest monthly average wind speed at approximately **20.83 km/h in July**.
- **Bengaluru** recorded the lowest average visibility, while **Jaipur** recorded the highest.
- Humidity and visibility showed a **moderate negative correlation of approximately -0.35**.
- Temperature and Heat Index showed a **very strong positive correlation of approximately 0.94**.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data analysis and application development |
| Pandas | Data cleaning, transformation and aggregation |
| NumPy | Numerical operations |
| Plotly | Interactive data visualization |
| Streamlit | Interactive dashboard |
| Matplotlib | Supporting visualizations |

---

