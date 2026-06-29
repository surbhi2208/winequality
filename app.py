from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

features = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        values = []

        for feature in features:
            values.append(float(request.form[feature]))

        prediction = model.predict([values])[0]

    return render_template(
        "index.html",
        features=features,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)
