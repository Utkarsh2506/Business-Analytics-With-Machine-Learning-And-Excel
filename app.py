from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained models
with open("wine_quality_red_model.pkl", "rb") as f:
    red_model = pickle.load(f)

with open("wine_quality_white_model.pkl", "rb") as f:
    white_model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form (matching dataset column names)
        features = []
        feature_names = [
            "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
            "pH", "sulphates", "alcohol"
        ]

        for name in feature_names:
            value = float(request.form[name])
            if value < 0:
                return render_template('index.html', error="❌ Input values cannot be negative!", user_inputs=request.form)
            features.append(value)


        # Get wine type: 0 = Red, 1 = White
        wine_type = request.form["wine_type"]
        model = red_model if wine_type == "red" else white_model

        # Convert input to numpy array
        input_data = np.array(features).reshape(1, -1)
        prediction = f"{model.predict(input_data)[0]:.2f}"

        return render_template('index.html', prediction=prediction, wine_type=wine_type, user_inputs=request.form)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
