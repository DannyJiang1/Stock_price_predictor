from predictor import Predictor


def main():

    # Train and save the model
    train_tickers = ["SPY"]  # Example training tickers
    predictor = Predictor(train_tickers)
    predictor.train_and_save_model(train_tickers)

    # Predict the closing price for a given ticker
    predict_ticker = "SPY"
    predictor.predict_price(predict_ticker)


if __name__ == "__main__":
    main()
