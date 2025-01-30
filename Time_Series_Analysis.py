import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Create a directory to store the plots if it doesn't exist
output_dir = './plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the dataset
file_path = "Q3_Q4_stock_data.csv"  # Ensure this file is in your repo
data = pd.read_csv(file_path)

# Convert Date column to datetime and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Prepare a list of unique companies
companies = data['Company Name'].unique()

# Iterate through each company
for company in companies:
    print(f"Processing company: {company}")

    # Filter data for the current company
    company_data = data[data['Company Name'] == company].sort_index()

    # Select the time series column (e.g., 'Price')
    time_series = company_data['Price']

    # Handle missing or invalid values
    time_series = time_series.fillna(method='ffill').fillna(method='bfill')
    time_series = time_series.replace([np.inf, -np.inf], np.nan).dropna()

    # Split the data into train and test sets
    train_size = int(len(time_series) * 0.8)
    train, test = time_series.iloc[:train_size], time_series.iloc[train_size:]

    # Train ARIMA model and generate forecast
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))

    # Combine train and test data
    combined_data = pd.concat([train, test])

    # Plot the visualizations
    plt.figure(figsize=(16, 10))

    # Subplot 1: Stock prices over time
    plt.subplot(2, 1, 1)
    plt.plot(combined_data.index, combined_data, label='Price', marker='o')
    plt.title(f"{company} Stock Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.grid()
    plt.legend()

    # Subplot 2: Train, Test, and Forecast
    plt.subplot(2, 1, 2)
    plt.plot(train.index, train, label='Train Data', color='blue')
    plt.plot(test.index, test, label='Test Data', color='orange')
    plt.plot(test.index, predictions, label='Forecast', color='green', linestyle='--')
    plt.title(f"{company} Stock Price with Forecast")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.grid()
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the plot as an image in the output directory
    plot_filename = f"{output_dir}/{company.replace(' ', '_')}_stock_price_plot.png"
    plt.savefig(plot_filename)
    print(f"Plot saved for {company} at {plot_filename}")

    # Show the plot
    plt.show()

    # Calculate evaluation metrics
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)

    print(f"{company} - MSE: {mse}, RMSE: {rmse}")
