from flask import Flask, jsonify, request
from predictor import Predictor

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    model_name = data["model_name"]
    ticker_to_predict = data["ticker_to_predict"]
    pred_start = data["pred_start"]
    pred_end = data["pred_end"]
    predictor = Predictor(model_name)
    predictor.predict_price(ticker_to_predict)


@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()
    model_name = data["model_name"]
    train_tickers = data["train_tickers"]
    predictor = Predictor(model_name)
    predictor.train_and_save_model(train_tickers)


if __name__ == "__main__":
    app.run(debug=True)
