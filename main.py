import pandas as pd
import requests
from datetime import datetime, timedelta
import time


import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load and prepare data
df = pd.read_csv('currency_usdt_rates.csv')
features = ['USD_USDT', 'CNY_USDT', 'Gold_USDT', 'Inflation_US', 'Interest_US', 'S&P500']  # Expand as needed
data = df[features].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences (e.g., 24-hour lookback)
def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(np.argmax(data[i + seq_length, :13]))  # Assume first 13 are currency rates
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, 24)

# Split data
train_size = int(len(X) * 0.7)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Build model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(24, len(features))))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(13, activation='softmax'))  # 13 currencies

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

# Predict best currency
latest_data = data_scaled[-24:]  # Last 24 hours
prediction = model.predict(latest_data.reshape(1, 24, len(features)))
best_currency_idx = np.argmax(prediction[0])
currencies = ['USD', 'CNY', 'EUR', 'JPY', 'GBP', 'INR', 'CAD', 'BRL', 'AUD', 'KRW', 'RUB', 'ZAR', 'NGN']
print(f"Best currency to invest in: {currencies[best_currency_idx]}")


# Binance API setup
binance_api_url = "https://api.binance.com/api/v3/ticker/price"
headers = {"X-MBX-APIKEY": "your_api_key_here"}  # Replace with your Binance API key
