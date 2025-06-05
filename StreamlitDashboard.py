import streamlit as st

from LSTMForDashboard import get_non_rolling_forecast_1
from SARIMAForDashboard import get_non_rolling_forecast_2
from XGBoostForDashboard import get_non_rolling_forecast_3
from RandomForestForDashboard import get_non_rolling_forecast_4

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")

st.title("📈 Sales Forecast Dashboard")
st.subheader("A Time Series Forecasting Project")

dataset_url = "https://github.com/ritvikmath/Time-Series-Analysis/blob/master/catfish.csv"
st.markdown(f"**Dataset:** [Download here]({dataset_url})")

st.markdown("---")

st.header("LSTM Model Forecast - 95.55% Max Accuracy")
st.write("Comparison of actual sales vs predicted sales using LSTM.")
fig_lstm = get_non_rolling_forecast_1()
st.pyplot(fig_lstm)

st.markdown("---")

st.header("SARIMA Model Forecast - 95.20% Max Accuracy")
st.write("Comparison of actual sales vs predicted sales using SARIMA.")
fig_sarima = get_non_rolling_forecast_2()
st.pyplot(fig_sarima)

st.markdown("---")

st.header("XGBoost Model Forecast - 95.78% Max Accuracy")
st.write("Comparison of actual sales vs predicted sales using XGBoost.")
fig_sarima = get_non_rolling_forecast_3()
st.pyplot(fig_sarima)

st.markdown("---")

st.header("Random Forest Model Forecast - 96.05% Max Accuracy")
st.write("Comparison of actual sales vs predicted sales using Random Forest.")
fig_sarima = get_non_rolling_forecast_4()
st.pyplot(fig_sarima)

st.markdown("---")
st.write("Project created by Will Barnard")


