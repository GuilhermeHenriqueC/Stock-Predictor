import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Fetch historical data for a stock (e.g., Apple)
ticker = "AAPL"
stock_data = yf.download(ticker, start="2025-01-03", end="2025-02-09")

# Save to CSV
stock_data.to_csv("stock_data.csv")




# Selecting only the 'Close' price for prediction
data = stock_data[['Close']].values

# Scaling data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Display first few rows
print(stock_data.head())

# Create features and target
X = stock_data.index.factorize()[0].reshape(-1,1)  # Converting dates to numerical
y = stock_data['Close'].values

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse}")

# Save to CSV
stock_data.to_csv("stock_data.csv")

# Print stock name
print(f"\nAnalyzing stock: {ticker}")

plt.figure(figsize=(12,6))
plt.plot(y_test, label="Actual Prices")
plt.plot(predictions, label="Predicted Prices", linestyle='dashed')
plt.legend()
plt.show()