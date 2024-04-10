# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 07:17:24 2024

@author: user
"""

import streamlit as st
from datetime import date
import pandas as pd

import yfinance as yf
from plotly import graph_objs as go
from statsmodels.tsa.arima.model import ARIMA

START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("HMY", "IHS", "JMIA", "DRD", "SSL", "MIXT", "SBSW", "KOS", "IMPUY", "NL",)
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

@st.cache_data(hash_funcs={complex: hash})
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# ARIMA Model
@st.cache_data
def arima_model(data, component):
    model = ARIMA(data[component], order=(5, 1, 0))
    results = model.fit()
    return results

arima_results_open = arima_model(data, 'Open')
arima_results_close = arima_model(data, 'Close')
arima_results_high = arima_model(data, 'High')
arima_results_low = arima_model(data, 'Low')

st.subheader('ARIMA Forecast')

# Forecast each component separately
arima_forecast_open = arima_results_open.forecast(periods=period)
arima_forecast_close = arima_results_close.forecast(periods=period)
arima_forecast_high = arima_results_high.forecast(periods=period)
arima_forecast_low = arima_results_low.forecast(periods=period)

# Convert arima_forecast to pandas DataFrame
forecast_index = pd.date_range(start=data['Date'].iloc[-1], periods=len(arima_forecast_open) + 1)[1:]
forecast_data = pd.DataFrame({
    'Date': forecast_index,
    'Open': arima_forecast_open,
    'Close': arima_forecast_close,
    'High': arima_forecast_high,
    'Low': arima_forecast_low
})

st.subheader('Forecasted Data')
st.write(forecast_data.tail(5))  # Display only the last five rows

def plot_forecasted_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Actual Close Price', mode='lines', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_data['Date'], y=forecast_data['Close'], name='Forecasted Close Price', mode='lines', line=dict(color='green', dash='dot')))
    fig.layout.update(title_text="Forecasted Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_forecasted_data()

def plot_forecasted_data_components():
    fig_open = go.Figure()
    fig_open.add_trace(go.Scatter(x=forecast_data['Date'], y=forecast_data['Open'], name='Forecasted Open Price', mode='lines', line=dict(color='orange')))
    fig_open.layout.update(title_text="Forecasted Open Price", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_open)

    fig_close = go.Figure()
    fig_close.add_trace(go.Scatter(x=forecast_data['Date'], y=forecast_data['Close'], name='Forecasted Close Price', mode='lines', line=dict(color='green')))
    fig_close.layout.update(title_text="Forecasted Close Price", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_close)

    fig_high = go.Figure()
    fig_high.add_trace(go.Scatter(x=forecast_data['Date'], y=forecast_data['High'], name='Forecasted High Price', mode='lines', line=dict(color='red')))
    fig_high.layout.update(title_text="Forecasted High Price", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_high)

    fig_low = go.Figure()
    fig_low.add_trace(go.Scatter(x=forecast_data['Date'], y=forecast_data['Low'], name='Forecasted Low Price', mode='lines', line=dict(color='blue')))
    fig_low.layout.update(title_text="Forecasted Low Price", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_low)

plot_forecasted_data_components()

# Calculate percentage of gains and losses
last_price = data['Close'].iloc[-1]
forecast_last_price = forecast_data['Close'].iloc[-1]
percentage_change = ((forecast_last_price - last_price) / last_price) * 100

st.subheader('Percentage of gains/losses')

# Check if it's a gain or a loss
if percentage_change >= 0:
    st.write(f"<span style='color:green'>{percentage_change:.2f}%</span>", unsafe_allow_html=True)
else:
    st.write(f"<span style='color:red'>{percentage_change:.2f}%</span>", unsafe_allow_html=True)