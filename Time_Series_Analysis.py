# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
file_path = "Q3_Q4_stock_data.csv"  # Ensure this file is in your repo
data = pd.read_csv(file_path)

# Convert the Date column to datetime and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Select the column to analyze (e.g., Price)
time_series = data['Price']

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(time_series, label='Price')
plt.title('Stock Price Time Series')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig(f"{output_dir}/time_series_plot.png")  # Save plot
plt.close()

# Decompose the time series into trend, seasonality, and residuals
decomposition = seasonal_decompose(time_series, model='additive', period=12)
decomposition.plot()
plt.savefig(f"{output_dir}/time_series_decomposition.png")  # Save decomposition plot
plt.close()

# Split data into train and test sets
train_size = int(len(time_series) * 0.8)
train, test = time_series.iloc[:train_size], time_series.iloc[train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

# Forecast
forecast = model_fit.forecast(steps=len(test))
forecast_index = test.index

# Plot the forecast vs actual values
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(forecast_index, forecast, label='Forecast', color='red')
plt.title('ARIMA Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig(f"{output_dir}/arima_forecast.png")  # Save forecast plot
plt.close()

# Evaluate the model
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
