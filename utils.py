from typing import List
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam, Adagrad, Nadam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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
    # print(val_predictions)
    # print("######################")
    # print(val_Y)
    val_rmse = np.sqrt(mean_squared_error(val_Y, val_predictions))
    return val_rmse


def print_stats(percent_changes: List[float], stock: str, start_date: str):

    print(percent_changes)

    gains = 0
    ng = 0
    losses = 0
    nl = 0
    totalR = 1

    for i in percent_changes:
        if i > 0:
            gains += i
            ng += 1
        else:
            losses += i
            nl += 1
        totalR = totalR * ((i / 100) + 1)

    totalR = round((totalR - 1) * 100, 2)

    if ng > 0:
        avgGain = gains / ng
        maxR = str(max(percent_changes))
    else:
        avgGain = 0
        maxR = "undefined"

    if nl > 0:
        avgLoss = losses / nl
        maxL = str(min(percent_changes))
        ratio = str(-avgGain / avgLoss)
    else:
        avgLoss = 0
        maxL = "undefined"
        ratio = "inf"

    if ng > 0 or nl > 0:
        battingAvg = ng / (ng + nl)
    else:
        battingAvg = 0

    print()
    print(
        "Results for "
        + stock
        + " going back to "
        + start_date
        + ", Sample size: "
        + str(ng + nl)
        + " trades"
    )
    print("Batting Avg: " + str(battingAvg))
    print("Gain/loss ratio: " + ratio)
    print("Average Gain: " + str(avgGain))
    print("Average Loss: " + str(avgLoss))
    print("Max Return: " + maxR)
    print("Max Loss: " + maxL)
    print(
        "Total return over " + str(ng + nl) + " trades: " + str(totalR) + "%"
    )
