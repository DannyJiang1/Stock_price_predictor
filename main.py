import os
from predictor import Predictor
import datetime as dt
import utils


def main():
    predictor = Predictor()

    # Train and save the model
    train_tickers = ["SPY"]  # Example training tickers
    predictor.train_and_save_model(train_tickers)

    # Predict the closing price for a given ticker
    predict_ticker = "QQQ"
    predictor.predict_price(predict_ticker)


if __name__ == "__main__":
    main()
