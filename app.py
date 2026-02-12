
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

model = pickle.load(open("rainfall_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = scaler.transform([features])
    prediction = model.predict(final_features)

    if prediction[0] == 1:
        result = "Suitable for Crop Cultivation"
    else:
        result = "Not Suitable for Crop Cultivation"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
