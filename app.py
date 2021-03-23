import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import joblib
import os


app = Flask(__name__)
model = joblib.load("model.joblib")


@app.route('/')
def home():
    return 'Heart Disease Risk Prediction'


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict(list(data.values()))
    # print(list(data.values())[0])
    # print(prediction)
    confidence = model.predict_proba([list(data.values())[0]])[:, 1]
    return jsonify(result=int(list(prediction)[0]), confidence_percentage=confidence[0]*100)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    # app.run(debug=True)
