import pickle
import os
import sys
import numpy as np
from flask import Flask, request, jsonify

# Add ML directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ML')))
from extract import run_checks
from elm import *
# from train import StackingEnsemble  # Import the class (adjust module name if different)

# Initialize Flask app
app = Flask(__name__)

# Path to the trained model
PKL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'stacking_ensemble.pkl')

def load_model(pkl_path):
    """Load a trained model from a .pkl file."""
    try:
        with open(pkl_path, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {pkl_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the model once at startup
MODEL = load_model(PKL_PATH)

def predict_phishing(url, model):
    """Send input to the model and get output."""
    try:
        if model is None:
            return {"error": "Model not loaded"}, 500

        # Prepare input using run_checks
        features = run_checks(url)
        if features is None:
            return {"error": "Failed to extract features"}, 400

        # Ensure features are in the correct shape (2D array for most models)
        if isinstance(features, list):
            features = np.array(features).reshape(1, -1)
            print(features)
        elif not isinstance(features, np.ndarray):
            return {"error": "Invalid feature format"}, 400

        # Get prediction
        prediction = model.predict(features)
        # Get probabilities if available
        probabilities = model.predict_proba(features) if hasattr(model, 'predict_proba') else None

        return {
            "url": url,
            "prediction": prediction[0].item(),  # Convert numpy type to Python type
            "probabilities": probabilities[0].tolist() if probabilities is not None else None
        }, 200
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}, 500

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict phishing for a given URL."""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "URL is required"}), 400

        url = data['url']
        result, status_code = predict_phishing(url, MODEL)
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({"error": f"Request failed: {e}"}), 500

if __name__ == "__main__":
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5050)