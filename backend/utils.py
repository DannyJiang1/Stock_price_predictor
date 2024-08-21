from typing import List
import numpy as np
import tensorflow as tf
import pandas_datareader as pdr
import yfinance as yf
import pandas as pd
import datetime as dt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam, Adagrad, Nadam
from sklearn.metrics import mean_squared_error


"""indicators: {
    rsi: {
        period: int,
        average_type: str,
    },
    bollinger_bands: {
        period: int,
        num_std_dev : int,
    },
    macd: {
        fast_period: int,
        slow_period: int,
        signal_period: int
    }
    emas: List[int]
}"""


def add_macroeconomic_indicators(df, indicators: dict, start, end):
    """Add macroeconomic indicators to the dataframe."""
    if "vix" in indicators and indicators["vix"]:
        vix_data = yf.download("^VIX", start=start, end=end)["Close"].rename(
            "VIX"
        )
        df["VIX"] = vix_data.reindex(df.index, method="ffill")
        print("LOG: Added volatility index (VIX) indicator. ")

    if "interest_rate" in indicators and indicators["interest_rate"]:
        ir_data = pdr.get_data_fred(
            indicators["interest_rate"], start=start, end=end
        ).rename(columns={indicators["interest_rate"]: "Interest Rate"})
        df["Interest Rate"] = ir_data.reindex(df.index, method="ffill")
        print("LOG: Added interest rate indicator.")

    if "unemployment_rate" in indicators and indicators["unemployment_rate"]:
        ur_data = pdr.get_data_fred(
            indicators["unemployment_rate"], start=start, end=end
        ).rename(
            columns={indicators["unemployment_rate"]: "Unemployment Rate"}
        )
        df["Unemployment Rate"] = ur_data.reindex(df.index, method="ffill")
        print("LOG: Added unemployment rate indicator.")

    if "consumer_sentiment" in indicators and indicators["consumer_sentiment"]:
        cs_data = pdr.get_data_fred(
            indicators["consumer_sentiment"], start=start, end=end
        ).rename(
            columns={indicators["consumer_sentiment"]: "Consumer Sentiment"}
        )
        df["Consumer Sentiment"] = cs_data.reindex(df.index, method="ffill")
        print("LOG: Added consumer sentiment indicator.")

    if "us_dollar_index" in indicators and indicators["us_dollar_index"]:
        usd_data = yf.download("DX-Y.NYB", start=start, end=end)[
            "Close"
        ].rename("US Dollar Index")
        df["US Dollar Index"] = usd_data.reindex(df.index, method="ffill")
        print("LOG: Added US dollar index indicator.")

    df.fillna(method="bfill", inplace=True)
    return df


def add_technical_indicators(df, indicators: dict):
    print("LOG: Adding technical indicators.")
    # RSI
    if "rsi" in indicators:
        period = indicators["rsi"]["period"]
        average_type = indicators["rsi"]["average_type"]
        rsi(df, period, average_type)

    # Bollinger Bands
    if "bollinger_bands" in indicators:
        period = indicators["bollinger_bands"]["period"]
        num_std_dev = indicators["bollinger_bands"]["num_std_dev"]
        bollinger_bands(df, period, num_std_dev)

    # MACD
    if "macd" in indicators:
        fast_period = indicators["macd"]["fast_period"]
        slow_period = indicators["macd"]["slow_period"]
        signal_period = indicators["macd"]["signal_period"]
        macd(df, fast_period, slow_period, signal_period)

        # EMAs

    if "emas" in indicators:
        for period in indicators["emas"]:
            df[f"EMA_{period}"] = (
                df["Close"].ewm(span=period, adjust=False).mean()
            )
    return df


def rsi(df, period=14, average_type="ema"):
    # Calculate the difference in price from the previous step
    delta = df["Close"].diff()

    # Make two series: one for gains and one for losses
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Calculate the exponential or simple moving average
    if average_type.lower() == "ema":
        average_gain = gain.ewm(com=(period - 1), min_periods=period).mean()
        average_loss = loss.ewm(com=(period - 1), min_periods=period).mean()
    elif average_type.lower() == "sma":
        average_gain = gain.rolling(window=period, min_periods=period).mean()
        average_loss = loss.rolling(window=period, min_periods=period).mean()

    # Calculate the RS
    RS = average_gain / average_loss

    # Calculate the RSI and add it to the DataFrame
    df["RSI"] = 100 - (100 / (1 + RS))
    print("LOG: Added RSI indicator.")


def bollinger_bands(df, period=20, num_std_dev=2):
    # Calculate the moving average (Middle Band)
    df["Middle Band"] = df["Close"].rolling(window=period).mean()

    # Calculate the standard deviation
    std_dev = df["Close"].rolling(window=period).std()

    # Calculate the Upper and Lower Bollinger Bands
    df["Upper Band"] = df["Middle Band"] + (std_dev * num_std_dev)
    df["Lower Band"] = df["Middle Band"] - (std_dev * num_std_dev)
    print("LOG: Added Bollinger Bands indicator.")


def macd(df, fast_period=12, slow_period=26, signal_period=9):
    # Calculate the fast and slow exponential moving averages
    ema_fast = df["Close"].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow_period, adjust=False).mean()

    # Calculate the MACD line
    df["MACD"] = ema_fast - ema_slow

    # Calculate the Signal line
    df["Signal Line"] = df["MACD"].ewm(span=signal_period, adjust=False).mean()
    print("LOG: Added MACD indicator.")


def emas(df, emas: List[int]):
    for ema in emas:
        df["Ema_" + str(ema)] = df["Close"].ewm(span=ema, adjust=False).mean()
        print(
            f"LOG: Added estimated moving average (EMA) for period of {ema}."
        )


def get_max_period(indicators):
    l = [0]
    if "rsi" in indicators:
        l.append(indicators["rsi"]["period"])
    if "bollinger_bands" in indicators:
        l.append(indicators["bollinger_bands"]["period"])
    if "macd" in indicators:
        l.append(indicators["macd"]["fast_period"])
        l.append(indicators["macd"]["slow_period"])
    if "emas" in indicators:
        l = l + indicators["emas"]
    max_period = max(l)
    if (
        "interest_rate" in indicators
        or "unemployment_rate" in indicators
        or "consumer_sentiment" in indicators
    ):
        max_period = max(max_period, 30)
    return max_period


# Function to download and preprocess stock data
def download_and_preprocess(ticker, start, end, indicators, dropna=True):
    print("LOG: Downloading data from Yahoo Finance.")
    df = yf.download(ticker, start, end)
    print("LOG: Download complete.")
    max_period = get_max_period(indicators)
    original_start = start
    if (end - start).days < max_period:
        start = end - dt.timedelta(days=max_period + 1)
        print(
            "LOG: Fetching extra data due to indicator periods longer than training period."
        )
        df = yf.download(ticker, start, end)
    df = add_technical_indicators(df, indicators)
    df = add_macroeconomic_indicators(df, indicators, start, end)
    df.drop(["Open", "High", "Low", "Adj Close"], axis=1, inplace=True)
    df["Next_Close"] = df["Close"].shift(-1)
    if dropna:
        df = df.dropna()
    df = df[df.index >= original_start]
    return df


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, keep_last=False):
    print("LOG: Transforming dataset into series format.")
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)

    # Separate features and labels
    features_df = df.drop(columns=["Next_Close"])
    labels_df = df[["Next_Close"]]

    cols, names = list(), list()

    # Input sequence (t-n+1, ... t)
    for i in range(n_in - 1, -1, -1):
        cols.append(features_df.shift(i))
        if i == 0:
            names += [
                (features_df.columns[j] + "(t)") for j in range(n_vars - 1)
            ]
        else:
            names += [
                (features_df.columns[j] + "(t-%d)" % i)
                for j in range(n_vars - 1)
            ]

    # Forecast sequence (t+1)
    for i in range(0, n_out):
        cols.append(labels_df.shift(-i))
        names += [(labels_df.columns[0] + "(t+%d)" % i)]

    # Concatenate all columns
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # Remove the last row before dropping NaN values
    last_row = agg.iloc[-1]

    if keep_last:
        last_row_df = pd.DataFrame([last_row], columns=agg.columns)
        if dropnan:
            agg.dropna(inplace=True)
        # This is done so that we still keep the last row despite it containing na for 'Next Close'
        # Since we need this row for the model to predict tomorrows price.
        agg = pd.concat([agg, last_row_df], ignore_index=True)
    elif dropnan:
        agg.dropna(inplace=True)

    # Separate features and labels
    n_obs = n_in * (n_vars - 1)
    features = agg.iloc[:, :n_obs]
    labels = agg.iloc[:, n_obs:]

    return features, labels


def build_model(model_spec):
    """model_spec:
    {
    layers:
        [
            units: int
        ],
    input_shape: tuple,
    optimizer: string,
    learning_rate: float,
    loss: string
    custom_loss : loss_function (if loss == "custom")
    }
    """
    # Build the LSTM model
    model = Sequential()
    for i, layer_units in enumerate(model_spec["layers"]):
        if i == 0:
            model.add(
                LSTM(
                    units=layer_units,
                    input_shape=model_spec["input_shape"],
                    return_sequences=(
                        True if len(model_spec["layers"]) > 1 else False
                    ),
                )
            )
        elif i != (len(model_spec["layers"]) - 1):
            model.add(LSTM(units=layer_units, return_sequences=True))
        else:
            model.add(LSTM(units=layer_units))

    model.add(Dense(units=1))

    if model_spec["loss"] == "custom":
        loss = model_spec["custom_loss"]
    else:
        loss = model_spec["loss"]

    if model_spec["optimizer"] == "Adam":
        model.compile(
            optimizer=Adam(learning_rate=model_spec["learning_rate"]),
            loss=loss,
        )
    elif model_spec["optimizer"] == "Adagrad":
        model.compile(
            optimizer=Adagrad(learning_rate=model_spec["learning_rate"]),
            loss=loss,
        )
    elif model_spec["optimizer"] == "Nadam":
        model.compile(
            optimizer=Nadam(learning_rate=model_spec["learning_rate"]),
            loss=loss,
        )
    else:
        raise Exception("optimizer must be 'Adam', 'Adagrad' or 'Nadam'")

    return model


def hyperparameter_tuning(
    optimizer_list,
    lr_list,
    batch_list,
    train_X,
    val_X,
    train_Y,
    val_Y,
):
    best_rmse = float("inf")
    best_params = {}
    for optimizer in optimizer_list:
        for learning_rate in lr_list:
            for batch_size in batch_list:
                # rmse_scores = []
                for _ in range(3):
                    print(
                        f"Training with optimizer={optimizer}, learning_rate={learning_rate}, batch_size={batch_size}"
                    )
                    rmse = train_and_evaluate(
                        train_X,
                        train_Y,
                        val_X,
                        val_Y,
                        [50, 20],
                        optimizer,
                        learning_rate,
                        batch_size,
                    )
                    print(f"RMSE: {rmse}")
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = {
                            "optimizer": optimizer,
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                        }

    print(f"Best RMSE: {best_rmse}")
    print(f"Best Hyperparameters: {best_params}")
    return best_rmse, best_params


def architecture_tuning(
    best_params, layers_list, train_X, val_X, train_Y, val_Y
):
    best_rmse = float("inf")
    best_architecture = []
    for layer_architecture in layers_list:
        for _ in range(3):
            print(f"Testing architecture: {layer_architecture}")
            rmse = train_and_evaluate(
                train_X,
                train_Y,
                val_X,
                val_Y,
                layer_architecture,
                best_params["optimizer"],
                best_params["learning_rate"],
                best_params["batch_size"],
            )
            print(f"RMSE: {rmse}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_architecture = layer_architecture

    print(f"Best RMSE: {best_rmse}")
    print(f"Best architecture: {best_architecture}")

    return best_architecture


def train_and_evaluate(
    train_X,
    train_Y,
    val_X,
    val_Y,
    layers,
    optimizer,
    learning_rate,
    batch_size,
    patience=5,
    max_epochs=100,
):
    model_spec = {
        "layers": layers,
        "input_shape": (train_X.shape[1], train_X.shape[2]),
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "loss": "mean_squared_error",
    }
    model = build_model(model_spec)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )

    model.fit(
        train_X,
        train_Y,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(val_X, val_Y),
        callbacks=[early_stopping],
        verbose=0,
    )

    val_predictions = model.predict(val_X)
    val_rmse = np.sqrt(mean_squared_error(val_Y, val_predictions))
    return val_rmse
