from predictor import Predictor
import datetime as dt


def main():

    # Train and save the model
    train_tickers = ["SPY"]  # Example training tickers
    predictor = Predictor(train_tickers)
    predictor.train_and_save_model(train_tickers)

    start_date_str = "2024-07-01"
    end_date_str = "2024-08-16"

    # Convert the date strings to datetime objects
    pred_start = dt.datetime.fromisoformat(start_date_str)
    pred_end = dt.datetime.fromisoformat(end_date_str)

    # Predict the closing price for a given ticker
    predict_ticker = "SPY"
    print(predictor.predict_price(predict_ticker))
    predictor.generate_graph(predict_ticker, pred_start, pred_end)


if __name__ == "__main__":
    main()
