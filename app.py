from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model/wine_cultivar_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['alcohol']),
        float(request.form['malic_acid']),
        float(request.form['ash']),
        float(request.form['alcalinity_of_ash']),
        float(request.form['magnesium']),
        float(request.form['flavanoids'])
    ]

    final_input = scaler.transform([features])
    prediction = model.predict(final_input)[0]

    return render_template("index.html", result=f"Cultivar {prediction+1}")

if __name__ == '__main__':
    app.run(debug=True)
