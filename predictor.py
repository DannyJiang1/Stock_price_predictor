"""Class that will be used to predict the closing price of stocks."""

import os
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

        # Download and preprocess data for each ticker
        df_list = [
            utils.download_and_preprocess(ticker, start, end, self.indicators)
            for ticker in train_tickers
        ]

        # Concatenate all dataframes
        df = pd.concat(df_list, axis=0)

        # Initialize separate scalers for features and the target
        self.features_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        # Scale the features (all columns except 'Next_Close')
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
        df_prepared = utils.series_to_supervised(
            df, n_in=self.n_input, n_out=1
        )

        # Split into input and outputs
        self.n_obs = self.n_input * len(df.columns)
        X, y = df_prepared.values[:, : self.n_obs], df_prepared.values[:, -1]

        X = X.reshape((X.shape[0], self.n_input, len(df.columns)))

        train_x, test_x, train_y, test_y = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        train_hypertune_x, val_x, train_hypertune_y, val_y = train_test_split(
            train_x, train_y, test_size=0.25, random_state=42
        )

        # Define the path to save/load the model
        self.model_path = "predictor_model.h5"

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

            # Architecture Tuning
            best_architecture = utils.architecture_tuning(
                best_params, layers_list, train_x, test_x, train_y, test_y
            )

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
                epochs=100,
                batch_size=best_params["batch_size"],
                validation_data=(test_x, test_y),
                verbose=2,
            )

            # Save the trained model
            final_model.save(self.model_path)

    def predict_price(self, predict_ticker):
        if not os.path.exists(self.model_path):
            print("No model present to use, training new model: ")
            self.train_and_save_model(predict_ticker)

        final_model = tf.keras.models.load_model(self.model_path)
        pred_start = dt.datetime(2021, 1, 1)
        pred_end = dt.datetime.now()

        # Download and preprocess data for the prediction ticker
        df_predict = utils.download_and_preprocess(
            predict_ticker, pred_start, pred_end, self.indicators
        )
        # Scale the features and target
        df_predict[self.feature_columns] = self.features_scaler.transform(
            df_predict[self.feature_columns]
        )
        df_predict[["Next_Close"]] = self.target_scaler.transform(
            df_predict[["Next_Close"]]
        )

        # Prepare the data for prediction
        df_predict_prepared = utils.series_to_supervised(
            df_predict, n_in=self.n_input, n_out=1
        )

        # Extract dates for the prediction ticker
        dates_predict = df_predict.index[self.n_input :]

        # Split into input and outputs
        predict_x, predict_y = (
            df_predict_prepared.values[:, : self.n_obs],
            df_predict_prepared.values[:, -1],
        )

        predict_x = predict_x.reshape(
            (predict_x.shape[0], self.n_input, len(self.feature_columns))
        )

        # Make predictions
        yhat = final_model.predict(predict_x)

        # Inverse transform to revert the scaling
        yhat_inverse = self.target_scaler.inverse_transform(
            yhat.reshape(-1, 1)
        )
        predict_y_inverse = self.target_scaler.inverse_transform(
            predict_y.reshape(-1, 1)
        )

        # Shift the predicted values by one day to align with actual values
        yhat_inverse_shifted = np.roll(yhat_inverse, 1)
        yhat_inverse_shifted[0] = (
            np.nan
        )  # Set the first value to NaN as it has no corresponding actual value

        # Calculate RMSE for the prediction ticker
        rmse = np.sqrt(
            mean_squared_error(predict_y_inverse[1:], yhat_inverse_shifted[1:])
        )

        print(f"Prediction RMSE for {predict_ticker}: %.3f" % rmse)

        # Plotting the final predictions
        plt.figure(figsize=(15, 5))
        plt.plot(dates_predict, predict_y_inverse, label="Actual Data")
        plt.plot(dates_predict, yhat_inverse_shifted, label="Predicted Data")
        plt.title(f"Comparison of Actual and Predicted - {predict_ticker}")
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.savefig("test.png")
