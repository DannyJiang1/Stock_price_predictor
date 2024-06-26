import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import tensorflow as tf
from strategies import add_technical_indicators
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Function to download and preprocess stock data
def download_and_preprocess(ticker, start, end, indicators):
    df = yf.download(ticker, start, end)
    df = add_technical_indicators(df, indicators)
    df.drop(["Open", "High", "Low", "Adj Close"], axis=1, inplace=True)
    df["Next_Close"] = df["Close"].shift(-1)
    df = df.dropna()
    return df


def custom_loss(y_true, y_pred):
    squared_error = tf.square(y_true - y_pred)
    overestimate_penalty = tf.where(
        y_pred > y_true, squared_error * 1, squared_error
    )
    return tf.reduce_mean(overestimate_penalty)


# Define the technical indicators
indicators = {
    "rsi": {"period": 14, "average_type": "ema"},
    "bollinger_bands": {"period": 20, "num_std_dev": 2},
    "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
    "emas": [12, 16, 50, 200],
}

# Define the tickers to train on and the prediction ticker
train_tickers = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "MARA",
    "NVDA",
    "COST",
]  # Add more tickers as needed
predict_ticker = "MARA"

start = dt.datetime(2020, 1, 1)
end = dt.datetime.now()

# Download and preprocess data for each ticker
df_list = [
    download_and_preprocess(ticker, start, end, indicators)
    for ticker in train_tickers
]

# Concatenate all dataframes
df = pd.concat(df_list, axis=0)

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
n_input = 2
df_prepared = series_to_supervised(df, n_in=n_input, n_out=1)

# Extract dates from the original data
dates = df.index[n_input:]

# Split into input and outputs
n_obs = n_input * len(df.columns)
X, y = df_prepared.values[:, :n_obs], df_prepared.values[:, -1]

# Reshape input to be 3D [samples, timesteps, features] as required by LSTM
X = X.reshape((X.shape[0], n_input, len(df.columns)))

# Split into training and testing sets
train_size = int(len(X) * 0.8)
train_X, test_X = X[:train_size], X[train_size:]
train_y, test_y = y[:train_size], y[train_size:]

model = Sequential()
model.add(
    LSTM(
        50,
        input_shape=(train_X.shape[1], train_X.shape[2]),
        return_sequences=True,
    )
)
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.005), loss=custom_loss)

# Train the model
history = model.fit(
    train_X,
    train_y,
    epochs=50,
    batch_size=64,
    validation_data=(test_X, test_y),
    verbose=2,
)

pred_start = dt.datetime(2024, 1, 1)


# Download and preprocess data for the prediction ticker
df_predict = download_and_preprocess(
    predict_ticker, pred_start, end, indicators
)

# Scale the features and target
df_predict[feature_columns] = features_scaler.transform(
    df_predict[feature_columns]
)
df_predict[["Next_Close"]] = target_scaler.transform(
    df_predict[["Next_Close"]]
)

# Prepare the data for prediction
df_predict_prepared = series_to_supervised(df_predict, n_in=n_input, n_out=1)

# Extract dates for the prediction ticker
dates_predict = df_predict.index[n_input:]

# Split into input and outputs
predict_X, predict_y = (
    df_predict_prepared.values[:, :n_obs],
    df_predict_prepared.values[:, -1],
)

# Reshape input to be 3D [samples, timesteps, features] as required by LSTM
predict_X = predict_X.reshape((predict_X.shape[0], n_input, len(df.columns)))

# Make predictions
yhat = model.predict(predict_X)

# Inverse transform to revert the scaling
yhat_inverse = target_scaler.inverse_transform(yhat.reshape(-1, 1))
predict_y_inverse = target_scaler.inverse_transform(predict_y.reshape(-1, 1))

# Shift the predicted values by one day to align with actual values
yhat_inverse_shifted = np.roll(yhat_inverse, 1)
yhat_inverse_shifted[0] = (
    np.nan
)  # Set the first value to NaN as it has no corresponding actual value

# Calculate RMSE for the prediction ticker, excluding the first shifted value
rmse = np.sqrt(
    mean_squared_error(predict_y_inverse[1:], yhat_inverse_shifted[1:])
)
print(f"Prediction RMSE for {predict_ticker}: %.3f" % rmse)

# Plotting the validation predictions
plt.figure(figsize=(15, 5))
plt.plot(dates_predict, predict_y_inverse, label="Actual Data")
plt.plot(dates_predict, yhat_inverse_shifted, label="Predicted Data")
plt.title(f"Comparison of Actual and Predicted - {predict_ticker}")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.savefig("test.png")
