import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Define the cryptocurrency symbol and download historical data using yfinance
cryptocurrency_symbol = 'BTC-USD'
data = yf.download(cryptocurrency_symbol, start='2022-01-01', end='2023-06-9')

# Extract the 'Close' price as the target variable
prices = data['Close'].values.reshape(-1, 1)

# Create a feature column representing the sequence of days
days = np.arange(len(prices)).reshape(-1, 1)

# Create and train the linear regression model
model = LinearRegression()
model.fit(days, prices)

# Predict prices for future days
future_days = np.arange(len(prices), len(prices) + 30).reshape(-1, 1)
predicted_prices = model.predict(future_days)

# Line Chart
plt.figure(figsize=(10, 6))
plt.plot(days, prices, label='Historical Prices')
plt.plot(future_days, predicted_prices, label='Predicted Prices')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title(f'{cryptocurrency_symbol} Price Prediction')
plt.legend()
plt.grid(True)
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(days, prices, label='Historical Prices', color='blue')
plt.plot(future_days, predicted_prices, label='Predicted Prices', color='red')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title(f'{cryptocurrency_symbol} Price Prediction')
plt.legend()
plt.grid(True)
plt.show()

# Bar Chart
plt.figure(figsize=(10, 6))
plt.bar(days.flatten(), prices.flatten(), label='Historical Prices', color='blue')
plt.plot(future_days, predicted_prices, label='Predicted Prices', color='red')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title(f'{cryptocurrency_symbol} Price Prediction')
plt.legend()
plt.grid(True)
plt.show()

# Area Chart
plt.figure(figsize=(10, 6))
plt.fill_between(days.flatten(), prices.flatten(), color='skyblue', alpha=0.3, label='Historical Prices')
plt.plot(future_days, predicted_prices, label='Predicted Prices', color='red')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title(f'{cryptocurrency_symbol} Price Prediction')
plt.legend()
plt.grid(True)
plt.show()

# KDE Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(prices.flatten(), color='blue', label='Historical Prices')
sns.kdeplot(predicted_prices.flatten(), color='red', label='Predicted Prices')
plt.xlabel('Price')
plt.ylabel('Density')
plt.title(f'{cryptocurrency_symbol} Price Prediction')
plt.legend()
plt.grid(True)
plt.show()
