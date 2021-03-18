import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import joblib


app = Flask(__name__)
model = joblib.load("model.joblib")


@app.route('/')
def home():
    return 'Heart Disease Risk Prediction'


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict(list(data.values()))
    return jsonify(result=int(list(prediction)[0]))


if __name__ == '__main__':
    app.run(debug=True)
