import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from pathlib import Path


# -----------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Electricity Consumption Analysis",
    page_icon="⚡",
    layout="wide"
)


# -----------------------------------
# 2. LOAD MODEL AND DATA
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():

    model = joblib.load(BASE_DIR / "models" / "best_model.pkl")
    features = joblib.load(BASE_DIR / "models" / "model_features.pkl")

    return model, features


@st.cache_data
def load_data():

    data = pd.read_csv(
        BASE_DIR / "data" / "processed" / "daily_consumption.csv"
    )

    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values("Date")

    return data


model, features = load_model()
data = load_data()


# -----------------------------------
# 3. SIDEBAR
# -----------------------------------

st.sidebar.title("⚡ Electricity Forecasting")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Consumption Analysis",
        "Forecasting",
        "Energy Recommendations",
        "About Project"
    ]
)

# -----------------------------------
# 4. DASHBOARD PAGE
# -----------------------------------

if page == "Dashboard":

    st.title("⚡ Electricity Consumption Dashboard")

    st.write(
        "Analyse historical electricity consumption "
        "and explore future consumption predictions."
    )

    total_consumption = data["Daily_Consumption_kWh"].sum()

    average_consumption = data["Daily_Consumption_kWh"].mean()

    maximum_consumption = data["Daily_Consumption_kWh"].max()

    minimum_consumption = data["Daily_Consumption_kWh"].min()


    # METRICS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Consumption",
        f"{total_consumption:,.2f} kWh"
    )

    col2.metric(
        "Average Daily Consumption",
        f"{average_consumption:.2f} kWh"
    )

    col3.metric(
        "Maximum Daily Consumption",
        f"{maximum_consumption:.2f} kWh"
    )

    col4.metric(
        "Minimum Daily Consumption",
        f"{minimum_consumption:.2f} kWh"
    )


    st.divider()


    # DAILY CONSUMPTION GRAPH

    st.subheader("Daily Electricity Consumption")

    fig = px.line(
        data,
        x="Date",
        y="Daily_Consumption_kWh",
        title="Daily Electricity Consumption Over Time"
    )

    st.plotly_chart(fig, use_container_width=True)


    # MONTHLY CONSUMPTION

    monthly_data = data.copy()

    monthly_data["Month"] = (
        monthly_data["Date"].dt.to_period("M").astype(str)
    )

    monthly_consumption = (
        monthly_data.groupby("Month")["Daily_Consumption_kWh"]
        .sum()
        .reset_index()
    )

    st.subheader("Monthly Electricity Consumption")

    fig = px.bar(
        monthly_consumption,
        x="Month",
        y="Daily_Consumption_kWh",
        title="Monthly Electricity Consumption"
    )

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------
# 5. CONSUMPTION ANALYSIS PAGE
# -----------------------------------

elif page == "Consumption Analysis":

    st.title("Consumption Analysis")

    st.write("Explore electricity consumption patterns.")

    data_copy = data.copy()

    data_copy["Day_of_Week"] = data_copy["Date"].dt.day_name()

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekday_data = (
        data_copy.groupby("Day_of_Week")["Daily_Consumption_kWh"]
        .mean()
        .reindex(weekday_order)
        .reset_index()
    )

    fig = px.bar(
        weekday_data,
        x="Day_of_Week",
        y="Daily_Consumption_kWh",
        title="Average Consumption by Day of the Week"
    )

    st.plotly_chart(fig, use_container_width=True)


    # HIGH CONSUMPTION DAYS

    st.subheader("Highest Consumption Days")

    highest_days = data.nlargest(
        10,
        "Daily_Consumption_kWh"
    )

    st.dataframe(highest_days, use_container_width=True)
        # -----------------------------------
    # ANOMALY DETECTION
    # -----------------------------------

    st.divider()

    st.subheader("⚠️ Unusual Consumption Detection")

    # Calculate quartiles

    Q1 = data["Daily_Consumption_kWh"].quantile(0.25)

    Q3 = data["Daily_Consumption_kWh"].quantile(0.75)

    IQR = Q3 - Q1

    # Calculate boundaries

    lower_bound = Q1 - (1.5 * IQR)

    upper_bound = Q3 + (1.5 * IQR)

    # Identify unusual consumption

    anomaly_data = data.copy()

    anomaly_data["Anomaly"] = anomaly_data[
        "Daily_Consumption_kWh"
    ] > upper_bound

    unusual_days = anomaly_data[
        anomaly_data["Anomaly"] == True
    ]

    # Display results

    col1, col2 = st.columns(2)

    col1.metric(
        "Unusually High Consumption Days",
        len(unusual_days)
    )

    col2.metric(
        "Upper Consumption Threshold",
        f"{upper_bound:.2f} kWh"
    )

    # Display unusual days

    st.write("### Days with Unusually High Consumption")

    if len(unusual_days) > 0:

        st.dataframe(
            unusual_days[
                ["Date", "Daily_Consumption_kWh"]
            ].sort_values(
                by="Daily_Consumption_kWh",
                ascending=False
            ),
            use_container_width=True
        )

    else:

        st.success("No unusually high consumption days detected.")

    # Visualise anomalies

    fig = px.scatter(
        anomaly_data,
        x="Date",
        y="Daily_Consumption_kWh",
        color="Anomaly",
        color_discrete_map={
            True: "red",
            False: "blue"
        },
        title="Electricity Consumption Anomaly Detection"
    )

    st.plotly_chart(fig, use_container_width=True)

    # LOW CONSUMPTION DAYS

    st.subheader("Lowest Consumption Days")

    lowest_days = data.nsmallest(
        10,
        "Daily_Consumption_kWh"
    )

    st.dataframe(lowest_days, use_container_width=True)


# -----------------------------------
# 6. FORECASTING PAGE
# -----------------------------------

elif page == "Forecasting":

    st.title("Future Electricity Consumption Forecast")

    st.write(
        "Use the trained machine learning model "
        "to estimate future electricity consumption."
    )

    forecast_days = st.selectbox(
        "Select Forecast Period",
        [7, 30]
    )

    tariff = st.number_input(
        "Electricity tariff (₹ per kWh)",
        min_value=0.0,
        value=8.0,
        step=0.5
    )


    # PREPARE HISTORICAL DATA

    history = data.set_index("Date")[
        "Daily_Consumption_kWh"
    ].sort_index()

    full_dates = pd.date_range(
        start=history.index.min(),
        end=history.index.max(),
        freq="D"
    )

    history = history.reindex(full_dates)

    history = history.interpolate(method="time", limit_area="inside")

    if history.iloc[-30:].isna().any():
        st.error(
            "The latest 30 days contain missing consumption values. "
            "Please check the processed dataset before forecasting."
        )
        st.stop()


    # BUILD FUTURE FORECAST RECURSIVELY

    future_predictions = []

    for i in range(forecast_days):

        next_date = history.index[-1] + pd.Timedelta(days=1)

        values = {
            "Year": next_date.year,
            "Month": next_date.month,
            "Day": next_date.day,
            "Day_of_Week": next_date.dayofweek,
            "Is_Weekend": int(next_date.dayofweek >= 5),

            "Consumption_Lag_1": history.iloc[-1],
            "Consumption_Lag_7": history.iloc[-7],
            "Consumption_Lag_30": history.iloc[-30],

            "Rolling_Mean_7": history.iloc[-7:].mean(),
            "Rolling_Mean_30": history.iloc[-30:].mean(),

            "Month_Sin": np.sin(2 * np.pi * next_date.month / 12),
            "Month_Cos": np.cos(2 * np.pi * next_date.month / 12)
        }

        input_data = pd.DataFrame(
            [values],
            columns=features
        )

        prediction = model.predict(input_data)[0]

        prediction = max(0, prediction)

        future_predictions.append({
            "Date": next_date,
            "Predicted_Consumption_kWh": prediction
        })

        # Add prediction to history so the next
        # forecast can use it as a lag value.

        history.loc[next_date] = prediction


    forecast_data = pd.DataFrame(future_predictions)


    # DISPLAY RESULTS

    st.subheader(f"Next {forecast_days} Days Forecast")

    st.dataframe(
        forecast_data.round(2),
        use_container_width=True
    )


    # FORECAST GRAPH

    fig = px.line(
        forecast_data,
        x="Date",
        y="Predicted_Consumption_kWh",
        title="Predicted Electricity Consumption"
    )

    st.plotly_chart(fig, use_container_width=True)


    # ESTIMATED COST

    total_predicted = forecast_data[
        "Predicted_Consumption_kWh"
    ].sum()

    estimated_cost = total_predicted * tariff

    col1, col2 = st.columns(2)

    col1.metric(
        "Predicted Total Consumption",
        f"{total_predicted:.2f} kWh"
    )

    col2.metric(
        "Estimated Electricity Cost",
        f"₹{estimated_cost:,.2f}"
    )

    st.info(
        "The electricity cost is an estimate based on "
        "the tariff entered above. Actual bills may include "
        "fixed charges, taxes, and other adjustments."
    )

# -----------------------------------
# 7. ENERGY RECOMMENDATIONS PAGE
# -----------------------------------

elif page == "Energy Recommendations":

    st.title("💡 Energy-Saving Recommendations")

    st.write(
        "Analyse your electricity consumption patterns "
        "and discover opportunities to reduce energy usage."
    )

    # Sort historical data

    analysis_data = data.sort_values("Date").copy()

    # Calculate average consumption

    overall_average = analysis_data[
        "Daily_Consumption_kWh"
    ].mean()

    # Get recent 7 days

    recent_data = analysis_data.tail(7)

    recent_average = recent_data[
        "Daily_Consumption_kWh"
    ].mean()

    # Calculate previous 7-day average

    previous_data = analysis_data.iloc[-14:-7]

    if len(previous_data) == 7:

        previous_average = previous_data[
            "Daily_Consumption_kWh"
        ].mean()

    else:

        previous_average = overall_average

    # Calculate weekday and weekend averages

    analysis_data["Day_of_Week"] = (
        analysis_data["Date"].dt.dayofweek
    )

    weekday_average = analysis_data[
        analysis_data["Day_of_Week"] < 5
    ]["Daily_Consumption_kWh"].mean()

    weekend_average = analysis_data[
        analysis_data["Day_of_Week"] >= 5
    ]["Daily_Consumption_kWh"].mean()

    # Calculate upper consumption threshold

    Q1 = analysis_data[
        "Daily_Consumption_kWh"
    ].quantile(0.25)

    Q3 = analysis_data[
        "Daily_Consumption_kWh"
    ].quantile(0.75)

    IQR = Q3 - Q1

    upper_bound = Q3 + (1.5 * IQR)

    # Display consumption metrics

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Overall Daily Average",
        f"{overall_average:.2f} kWh"
    )

    col2.metric(
        "Recent 7-Day Average",
        f"{recent_average:.2f} kWh",
        delta=f"{recent_average - overall_average:.2f} kWh"
    )

    col3.metric(
        "Previous 7-Day Average",
        f"{previous_average:.2f} kWh"
    )

    st.divider()

    # GENERATE RECOMMENDATIONS

    st.subheader("Your Personalised Recommendations")

    recommendations = []

    # Recommendation 1: Recent consumption

    if recent_average > overall_average:

        increase = (
            (recent_average - overall_average)
            / overall_average
        ) * 100

        recommendations.append(
            (
                "⚠️ High Recent Consumption",
                f"Your average consumption over the last "
                f"7 days is {increase:.1f}% higher than "
                f"your historical daily average. "
                f"Check whether appliances such as air "
                f"conditioners, heaters or water heaters "
                f"are being used more frequently."
            )
        )

    else:

        recommendations.append(
            (
                "✅ Good Consumption Pattern",
                "Your recent 7-day average is not higher "
                "than your historical daily average. "
                "Continue monitoring your electricity usage."
            )
        )

    # Recommendation 2: Weekly increase

    if recent_average > previous_average:

        weekly_increase = (
            (recent_average - previous_average)
            / previous_average
        ) * 100 if previous_average > 0 else 0

        recommendations.append(
            (
                "📈 Weekly Consumption Increase",
                f"Your consumption has increased by "
                f"{weekly_increase:.1f}% compared with "
                f"the previous 7 days. Review recent "
                f"changes in appliance usage."
            )
        )

    else:

        recommendations.append(
            (
                "📉 Weekly Consumption Improvement",
                "Your recent consumption is lower than "
                "the previous week's average. "
                "Keep following your current energy habits."
            )
        )

    # Recommendation 3: Weekend consumption

    if weekend_average > weekday_average:

        recommendations.append(
            (
                "🏠 Monitor Weekend Usage",
                "Your average weekend consumption is "
                "higher than your weekday consumption. "
                "Consider switching off unnecessary "
                "appliances when they are not in use."
            )
        )

    else:

        recommendations.append(
            (
                "✅ Weekend Consumption",
                "Your weekend consumption is not higher "
                "than your weekday consumption."
            )
        )

    # Recommendation 4: Unusual consumption

    unusual_days = analysis_data[
        analysis_data["Daily_Consumption_kWh"] > upper_bound
    ]

    if len(unusual_days) > 0:

        recommendations.append(
            (
                "🚨 Unusual Consumption Detected",
                f"We identified {len(unusual_days)} days "
                f"with unusually high electricity usage. "
                f"Check whether high-power appliances "
                f"were operating for longer than usual."
            )
        )

    else:

        recommendations.append(
            (
                "✅ No Unusual Consumption",
                "No unusually high consumption days "
                "were detected using the IQR method."
            )
        )

    # DISPLAY RECOMMENDATIONS

    for title, message in recommendations:

        with st.container(border=True):

            st.markdown(f"### {title}")

            st.write(message)

    # CONSUMPTION COMPARISON

    st.divider()

    st.subheader("Weekday vs Weekend Consumption")

    comparison_data = pd.DataFrame({
        "Period": ["Weekdays", "Weekends"],
        "Average_Consumption": [
            weekday_average,
            weekend_average
        ]
    })

    fig = px.bar(
        comparison_data,
        x="Period",
        y="Average_Consumption",
        title="Average Electricity Consumption"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
# -----------------------------------
# 7. ABOUT PROJECT PAGE
# -----------------------------------

elif page == "About Project":

    st.title("About This Project")

    st.write("""
    ### Electricity Consumption Analysis and Forecasting

    This project uses historical electricity consumption
    data and machine learning to analyse usage patterns
    and predict future electricity consumption.

    ### Technologies Used

    - Python
    - Pandas
    - NumPy
    - Matplotlib
    - Plotly
    - Scikit-learn
    - XGBoost
    - Streamlit

    ### Machine Learning

    The project compares multiple regression algorithms
    to identify the best-performing forecasting model.

    ### Main Features

    - Historical consumption analysis
    - Daily and monthly visualisations
    - Electricity consumption forecasting
    - Estimated electricity costs
    - Consumption pattern analysis
    """)