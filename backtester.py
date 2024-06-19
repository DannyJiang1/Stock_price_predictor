import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
from strategies import add_technical_indicators
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

now = dt.datetime.now()
stock = input("Enter Ticker: ")
print(stock)

start = dt.datetime(2022, 1, 1)
df = yf.download(stock, start, now)

# fig, ax = plt.subplots()
# mpf.plot(df, ax=ax, type="candle", style="charles", ylabel="Price ($)")
# ax.set_title(f"Candlestick chart for {stock}")
# plt.show()

# st_lt_ema(stock, df, [12, 16], [50, 200])

indicators = {
    "rsi": {
        "period": 14,
        "average_type": "ema",
    },
    "bollinger_bands": {
        "period": 20,
        "num_std_dev": 2,
    },
    "macd": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
    },
    "emas": [12, 16, 50, 200],
}

df = add_technical_indicators(df, indicators)
df.drop(["Open", "High", "Low", "Adj Close"], axis=1, inplace=True)
df["Next_Close"] = df["Close"].shift(-1)
df = df.dropna()

# Initialize separate scalers for features and the target
features_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Scale the features (all columns except 'Next_Close')
feature_columns = [col for col in df.columns if col != "Next_Close"]
df[feature_columns] = features_scaler.fit_transform(df[feature_columns])

# Scale the target ('Next_Close')
df[["Next_Close"]] = target_scaler.fit_transform(df[["Next_Close"]])


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(df.columns[j] + "(t-%d)" % i) for j in range(n_vars)]
    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(df.columns[j] + "(t)") for j in range(n_vars)]
        else:
            names += [(df.columns[j] + "(t+%d)" % i) for j in range(n_vars)]
    # Concatenate all columns
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# Assuming you use 3 days of past data to predict the next close
n_input = 3
df_prepared = series_to_supervised(df, n_in=n_input)

# Split into input and outputs
n_obs = n_input * len(df.columns)
train_X, train_y = (
    df_prepared.values[:, :n_obs],
    df_prepared.values[:, -1],
)  # assuming the last column is 'Next_Close(t)'
test_X, test_y = df_prepared.values[:, :n_obs], df_prepared.values[:, -1]

# Reshape input to be 3D [samples, timesteps, features] as required by LSTM
train_X = train_X.reshape((train_X.shape[0], n_input, len(df.columns)))
test_X = test_X.reshape((test_X.shape[0], n_input, len(df.columns)))

model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
history = model.fit(
    train_X,
    train_y,
    epochs=50,
    batch_size=72,
    validation_data=(test_X, test_y),
    verbose=2,
)

# Make a prediction
yhat = model.predict(test_X)

# Inverse transform to revert the scaling
yhat_inverse = target_scaler.inverse_transform(yhat.reshape(-1, 1))
test_y_inverse = target_scaler.inverse_transform(test_y.reshape(-1, 1))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_y_inverse, yhat_inverse))
print("Test RMSE: %.3f" % rmse)


# Predictions on training data
train_pred = model.predict(train_X)

# Predictions on validation (test) data
test_pred = model.predict(test_X)

# Inverse transform predictions
train_pred_inverse = target_scaler.inverse_transform(train_pred)
test_pred_inverse = target_scaler.inverse_transform(test_pred)

# Inverse transform actual values
train_y_inverse = target_scaler.inverse_transform(train_y.reshape(-1, 1))
test_y_inverse = target_scaler.inverse_transform(test_y.reshape(-1, 1))

import matplotlib.pyplot as plt

# Plotting the training predictions
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(train_y_inverse, label="Actual Training Data")
plt.plot(train_pred_inverse, label="Predicted Training Data")
plt.title("Comparison of Actual and Predicted - Training Data")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()

# Plotting the validation predictions
plt.subplot(1, 2, 2)
plt.plot(test_y_inverse, label="Actual Validation Data")
plt.plot(test_pred_inverse, label="Predicted Validation Data")
plt.title("Comparison of Actual and Predicted - Validation Data")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()

plt.show()
