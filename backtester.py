import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from strategies import add_technical_indicators, add_macroeconomic_indicators
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from utils import hyperparameter_tuning, architecture_tuning, build_model


# Function to download and preprocess stock data
def download_and_preprocess(ticker, start, end, indicators):
    df = yf.download(ticker, start, end)
    df = add_technical_indicators(df, indicators)
    df = add_macroeconomic_indicators(df, indicators, start, end)
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
    "vix": True,
    "interest_rate": "EFFR",
    "unemployment_rate": "UNRATE",
    "consumer_sentiment": "UMCSENT",
    "us_dollar_index": True,
}

# Define the tickers to train on and the prediction ticker
train_tickers = ["SPY"]  # Add more tickers as needed
predict_ticker = "SPY"

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


# Increase the number of past timesteps to use for prediction
n_input = 5
df_prepared = series_to_supervised(df, n_in=n_input, n_out=1)

# Extract dates from the original data
dates = df.index[n_input:]

# Split into input and outputs
n_obs = n_input * len(df.columns)
X, y = df_prepared.values[:, :n_obs], df_prepared.values[:, -1]

# Reshape input to be 3D [samples, timesteps, features] as required by LSTM
X = X.reshape((X.shape[0], n_input, len(df.columns)))

optimizers = ["Adam"]
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64]


train_X, test_X, train_Y, test_Y = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_X, val_X, train_Y, val_Y = train_test_split(
    train_X, train_Y, test_size=0.25, random_state=42
)

# best_rmse, best_params = hyperparameter_tuning(
#     optimizers, learning_rates, batch_sizes, train_X, val_X, train_Y, val_Y
# )

best_params = {
    "optimizer": "Adam",
    "learning_rate": 0.1,
    "batch_size": 32,
}

layers_list = [[50], [50, 20]]

best_architecture = architecture_tuning(
    best_params, layers_list, train_X, test_X, train_Y, test_Y
)

model_spec = {
    "layers": best_architecture,
    "input_shape": (train_X.shape[1], train_X.shape[2]),
    "optimizer": best_params["optimizer"],
    "learning_rate": best_params["learning_rate"],
    "loss": "mean_squared_error",
}

final_model = build_model(model_spec)
history = final_model.fit(
    train_X,
    train_Y,
    epochs=100,
    batch_size=best_params["batch_size"],
    validation_data=(test_X, test_Y),
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
yhat = final_model.predict(predict_X)

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
