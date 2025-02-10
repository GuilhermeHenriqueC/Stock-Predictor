import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Fetch historical data for a stock (e.g., Apple)
ticker = "AAPL"
stock_data = yf.download(ticker, start="2015-01-01", end="2024-01-01")

# Save to CSV
stock_data.to_csv("stock_data.csv")




# Selecting only the 'Close' price for prediction
data = stock_data[['Close']].values

# Scaling data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Display first few rows
print(stock_data.head())

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prepare time series data
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60  # 60 days of data to predict next day
X, y = create_sequences(scaled_data)

# Split into train/test
X_train, X_test = X[:-200], X[-200:]
y_train, y_test = y[:-200], y[-200:]

# Reshape for LSTM
X_train = X_train.reshape(-1, seq_length, 1)
X_test = X_test.reshape(-1, seq_length, 1)

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Compile & Train
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=16, epochs=50)

# Predict
predictions = model.predict(X_test)