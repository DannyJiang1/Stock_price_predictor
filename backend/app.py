from flask import Flask, jsonify, request, send_file
from predictor import Predictor
import datetime as dt

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    print("received predict")
    data = request.get_json()
    model_ticker = data["model_ticker"]
    ticker_to_predict = data["ticker_to_predict"]
    predictor = Predictor(model_ticker)
    prediction = predictor.predict_price(ticker_to_predict)
    return jsonify({"prediction": prediction})


@app.route("/graph", methods=["POST"])
def get_graph():
    print("received graph")
    data = request.get_json()
    model_ticker = data["model_ticker"]
    ticker_to_predict = data["ticker_to_predict"]
    start_date = dt.datetime.fromisoformat(data["start_date"])
    end_date = dt.datetime.fromisoformat(data["end_date"])
    predictor = Predictor(model_ticker)

    graph_path = predictor.generate_graph(
        ticker_to_predict, start_date, end_date
    )
    return send_file(graph_path, as_attachment=True)


@app.route("/train", methods=["POST"])
def train():
    print("received train")
    data = request.get_json()
    train_tickers = data["train_tickers"]
    predictor = Predictor(train_tickers)
    predictor.train_and_save_model(train_tickers)
    return jsonify({"train_status": "complete"})


if __name__ == "__main__":
    app.run(debug=True)
