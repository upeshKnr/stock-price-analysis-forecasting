import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Stock Price Analysis & Forecasting",
    page_icon="📈",
    layout="wide"
)

# ------------------ LOAD CSS ------------------
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/apple_stock.csv")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    return df

df = load_data()

# ------------------ TITLE ------------------
st.title("Apple Stock Analysis & Forecasting Dashboard 📈")
st.write("Interactive visualization and forecasting of Apple stock prices.")


# ------------------ SIDEBAR ------------------
st.sidebar.header("Filters")

start_date = st.sidebar.date_input("Start Date", df['Date'].min())
end_date = st.sidebar.date_input("End Date", df['Date'].max())

filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) &
                 (df['Date'] <= pd.to_datetime(end_date))]


st.subheader("Closing Price Trend")

fig = px.line(filtered_df, x="Date", y="Close",
              title="Apple Closing Price Over Time")
st.plotly_chart(fig, use_container_width=True)


# ------------------ LOAD MODELS ------------------
arima_model = joblib.load("models/arima_model.pkl")
lstm_model = load_model("models/lstm_model.keras")
scaler = joblib.load("models/lstm_scaler.pkl")


st.subheader("Moving Averages")

filtered_df['MA20'] = filtered_df['Close'].rolling(20).mean()
filtered_df['MA50'] = filtered_df['Close'].rolling(50).mean()

fig_ma = px.line(filtered_df, x="Date", y=["Close", "MA20", "MA50"])
st.plotly_chart(fig_ma, use_container_width=True)


st.subheader("Model Insights")

st.markdown("""
- **ARIMA** works well for linear trends.
- **LSTM** captures non-linear patterns.
- LSTM achieved lowest RMSE → best performer.
""")


#  forecasting Section
st.subheader("Forecasting Future Prices")

n_days = st.slider("Select number of days to forecast", 1, 60, 30)

last_60_days = df['Close'][-60:].values
last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))

temp_input = list(last_60_days_scaled.flatten())
lstm_predictions = []

for i in range(n_days):
    x_input = np.array(temp_input[-60:])
    x_input = x_input.reshape(1, 60, 1)

    yhat = lstm_model.predict(x_input, verbose=0)[0][0]
    lstm_predictions.append(yhat)
    temp_input.append(yhat)

# Inverse scale
lstm_predictions = scaler.inverse_transform(
    np.array(lstm_predictions).reshape(-1, 1)
).flatten()

forecast_dates = pd.date_range(
    start=df['Date'].max() + pd.Timedelta(days=1),
    periods=n_days
)

forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Forecasted Close': lstm_predictions
})

fig_forecast = px.line(
    forecast_df,
    x='Date',
    y='Forecasted Close',
    title="LSTM Forecasted Closing Prices"
)

st.plotly_chart(fig_forecast, use_container_width=True)

st.markdown("**Note:** LSTM model used for forecasting due to superior performance.")
