import numpy as np
import joblib
from flask import Flask, render_template, request
import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
model = joblib.load(os.path.join(os.path.dirname(__file__), "cancerdata.pkl"))

FEATURES = [
    "mean_texture", "mean_area", "mean_compactness", "mean_concave_pts",
    "radius_error", "area_error", "smoothness_error", "compactness_error",
    "fractal_dim_error", "worst_texture", "worst_perimeter",
    "worst_smoothness", "worst_concavity", "worst_concave_pts",
    "worst_symmetry"
]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [float(request.form[f]) for f in FEATURES]
        X = np.array([values])

        pred = model.predict(X)[0]
        prediction = "Benign" if pred == 1 else "Malignant"

        return render_template("index.html", prediction=prediction)
    except Exception as e:
        return render_template("index.html", prediction=None, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)