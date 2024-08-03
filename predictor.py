"""Class that will be used to predict the closing price of stocks."""

import os
import json
import datetime as dt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import utils


class Predictor:
    """Predict class."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.model_dir = self.model_name
        self.model_path = os.path.join(self.model_dir, "predictor_model.h5")
        self.info_path = os.path.join(self.model_dir, "model_info.json")

    def train_and_save_model(self, train_tickers_in=["SPY"]):
        """Train and save a new model given training and predicting tickers."""
        # Define the tickers to train on and the prediction ticker
        train_tickers = train_tickers_in
        self.indicators = {
            "rsi": {"period": 14, "average_type": "ema"},
            # "bollinger_bands": {"period": 20, "num_std_dev": 2},
            "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            "emas": [12, 16, 50, 200],
            "vix": True,
            "interest_rate": "EFFR",
            "unemployment_rate": "UNRATE",
            "consumer_sentiment": "UMCSENT",
            "us_dollar_index": True,
        }

        start = dt.datetime(2006, 1, 1)
        end = dt.datetime.now()

        optimizers = ["Adam", "Adagrad", "Nadam"]
        learning_rates = [0.001, 0.01, 0.1]
        batch_sizes = [4, 8, 16]

        layers_list = [[10], [30], [50], [100], [150], [200]]

        df_list = [
            utils.download_and_preprocess(ticker, start, end, self.indicators)
            for ticker in train_tickers
        ]

        # Concatenate all dataframes
        df = pd.concat(df_list, axis=0)

        # Initialize separate scalers for features and the target
        self.features_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        self.feature_columns = [
            col for col in df.columns if col != "Next_Close"
        ]
        df[self.feature_columns] = self.features_scaler.fit_transform(
            df[self.feature_columns]
        )

        # Scale the target ('Next_Close')
        df[["Next_Close"]] = self.target_scaler.fit_transform(
            df[["Next_Close"]]
        )

        # Increase the number of past timesteps to use for prediction
        self.n_input = 5
        X, y = utils.series_to_supervised(df, n_in=self.n_input, n_out=1)

        # Split into input and outputs
        self.n_obs = self.n_input * len(df.columns)
        X, y = X.values[:, : self.n_obs], y.values[:, 0]

        X = X.reshape((X.shape[0], self.n_input, len(df.columns) - 1))

        train_x, test_x, train_y, test_y = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        train_hypertune_x, val_x, train_hypertune_y, val_y = train_test_split(
            train_x, train_y, test_size=0.25, random_state=42
        )

        # Create the directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Check if the model already exists
        if os.path.exists(self.model_path):
            # Load the saved model
            print("predictor_model.h5 already exists.")
        else:
            # Hyperparamter Tuning
            _, best_params = utils.hyperparameter_tuning(
                optimizers,
                learning_rates,
                batch_sizes,
                train_hypertune_x,
                val_x,
                train_hypertune_y,
                val_y,
            )

            # best_params = {
            #     "optimizer": "Adam",
            #     "learning_rate": 0.01,
            #     "batch_size": 4,
            # }

            # Architecture Tuning
            best_architecture = utils.architecture_tuning(
                best_params, layers_list, train_x, test_x, train_y, test_y
            )

            # best_architecture = [50]

            model_spec = {
                "layers": best_architecture,
                "input_shape": (train_x.shape[1], train_x.shape[2]),
                "optimizer": best_params["optimizer"],
                "learning_rate": best_params["learning_rate"],
                "loss": "mean_squared_error",
            }

            # Final model with tuned hyperparameters and architecture
            final_model = utils.build_model(model_spec)
            final_model.fit(
                train_x,
                train_y,
                epochs=10,
                batch_size=best_params["batch_size"],
                validation_data=(test_x, test_y),
                verbose=2,
            )

            # Save the trained model
            final_model.save(self.model_path)

            model_info = {
                "indicators": self.indicators,
                "features_scaler": self.features_scaler.data_max_.tolist(),
                "target_scaler": self.target_scaler.data_max_.tolist(),
                "feature_columns": self.feature_columns,
                "n_input": self.n_input,
                "n_obs": self.n_obs,
            }

            with open(self.info_path, "w") as f:
                json.dump(model_info, f, indent=4)

    def predict_price(self, predict_ticker, graph=True):
        if not os.path.exists(self.model_path):
            print("No model present to use, training new model: ")
            self.train_and_save_model()

        final_model = tf.keras.models.load_model(self.model_path)
        pred_start = dt.datetime(2021, 1, 1)
        end = dt.datetime.now()
        indicators = self.indicators

        # Download and preprocess data for the prediction ticker
        df = utils.download_and_preprocess(
            predict_ticker, pred_start, end, indicators, dropna=False
        )
        print(df)

        df[self.feature_columns] = self.features_scaler.transform(
            df[self.feature_columns]
        )

        df[["Next_Close"]] = self.target_scaler.transform(df[["Next_Close"]])

        X_predict, y_predict = utils.series_to_supervised(
            df, n_in=self.n_input, n_out=1, keep_last=True
        )

        dates_predict = X_predict.index[:-1]

        # Split into input and outputs
        self.n_obs = self.n_input * (len(df.columns) - 1)

        X_predict, y_predict = (
            X_predict.values[:, : self.n_obs],
            y_predict.values[:, 0],
        )

        X_predict = X_predict.reshape(
            (X_predict.shape[0], self.n_input, len(df.columns) - 1)
        )

        # Make predictions
        yhat = final_model.predict(X_predict)

        # Inverse transform to revert the scaling
        yhat_inverse = self.target_scaler.inverse_transform(
            yhat.reshape(-1, 1)
        )

        tomorrow_price_prediction = yhat_inverse[-1]

        if graph and len(yhat_inverse) > 1:
            y_predict = y_predict[:-1]
            y_predict_inverse = self.target_scaler.inverse_transform(
                y_predict.reshape(-1, 1)
            )
            # Shift the predicted values by one day to align with actual values
            yhat_inverse_shifted = yhat_inverse[:-1]

            # Calculate RMSE for the prediction ticker
            rmse = np.sqrt(
                mean_squared_error(y_predict_inverse, yhat_inverse_shifted)
            )

            print(f"Prediction RMSE for {predict_ticker}: %.3f" % rmse)

            # return dates_predict, y_predict_inverse, yhat_inverse_shifted

            # Plotting the final predictions
            plt.figure(figsize=(15, 5))
            plt.plot(dates_predict, y_predict_inverse, label="Actual Data")
            plt.plot(
                dates_predict, yhat_inverse_shifted, label="Predicted Data"
            )
            plt.title(f"Comparison of Actual and Predicted - {predict_ticker}")
            plt.xlabel("Time")
            plt.ylabel("Stock Price")
            plt.legend()
            plt.savefig(f"{predict_ticker}.png")
