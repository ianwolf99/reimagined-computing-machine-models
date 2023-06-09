import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Define the cryptocurrency symbol and download historical data using yfinance
cryptocurrency_symbol = 'BTC-USD'
data = yf.download(cryptocurrency_symbol, start='2018-01-01', end='2023-06-09')

# Clean the data by removing missing values
data_cleaned = data.dropna()
print(data_cleaned)

# Extract the 'Close' price as the target variable
prices = data_cleaned['Close'].values.reshape(-1, 1)

# Create a feature column representing the sequence of days
days = np.arange(len(prices)).reshape(-1, 1)

# Create and train the linear regression model using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
model.fit(days, prices, epochs=100, verbose=0)

# Predict prices for future days
future_days = np.arange(len(prices), len(prices) + 30).reshape(-1, 1)
predicted_prices = model.predict(future_days)

# Save the cleaned data to a CSV file
data_cleaned.to_csv('cleaned_data.csv', index=False)

# Create subplots for multiple charts
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Line Chart
axs[0, 0].plot(days, prices, label='Historical Prices')
axs[0, 0].plot(future_days, predicted_prices, label='Predicted Prices')
axs[0, 0].set_xlabel('Days')
axs[0, 0].set_ylabel('Price')
axs[0, 0].set_title(f'{cryptocurrency_symbol} Price Prediction - Line Chart')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Scatter Plot
axs[0, 1].scatter(days, prices, label='Historical Prices', color='blue')
axs[0, 1].plot(future_days, predicted_prices, label='Predicted Prices', color='red')
axs[0, 1].set_xlabel('Days')
axs[0, 1].set_ylabel('Price')
axs[0, 1].set_title(f'{cryptocurrency_symbol} Price Prediction - Scatter Plot')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Bar Chart
axs[1, 0].bar(days.flatten(), prices.flatten(), label='Historical Prices', color='blue')
axs[1, 0].plot(future_days, predicted_prices, label='Predicted Prices', color='red')
axs[1, 0].set_xlabel('Days')
axs[1, 0].set_ylabel('Price')
axs[1, 0].set_title(f'{cryptocurrency_symbol} Price Prediction - Bar Chart')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Area Chart
axs[1, 1].fill_between(days.flatten(), prices.flatten(), color='skyblue', alpha=0.3, label='Historical Prices')
axs[1, 1].plot(future_days, predicted_prices, label='Predicted Prices', color='red')
axs[1, 1].set_xlabel('Days')
axs[1, 1].set_ylabel('Price')
axs[1, 1].set_title(f'{cryptocurrency_symbol} Price Prediction - Area Chart')
axs[1, 1].legend()
axs[1, 1].grid(True)

# Adjust the spacing between subplots
plt.tight_layout()

# KDE Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(prices.flatten(), color='blue', label='Historical Prices')
sns.kdeplot(predicted_prices.flatten(), color='red', label='Predicted Prices')
plt.xlabel('Price')
plt.ylabel('Density')
plt.title(f'{cryptocurrency_symbol} Price Prediction - KDE Plot')
plt.legend()
plt.grid(True)

# Display all the plots
plt.show()
