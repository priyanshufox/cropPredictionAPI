from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 
# Load model & encoder
model = joblib.load("crop_recommendation_hybrid.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")


def rule_based_filter(N, P, K, temperature, humidity, ph, rainfall):
    """
    This function applies simple rule-based logic to suggest suitable crops.
    """
    suggestions = []

    # Example rule-based conditions (Modify based on domain knowledge)
    if temperature > 25 and humidity > 80 and rainfall > 200:
        suggestions.append("rice")

    if N > 50 and P > 30 and K > 30 and temperature > 18 and ph > 5.5:
        suggestions.append("maize")

    if ph < 5.5:
        suggestions.append("banana")

    return suggestions if suggestions else ["No specific recommendation"]


def suggest_parameter_modifications(N, P, K, temperature, humidity, ph, rainfall, ml_crop, rule_suggestions):
    """
    Suggest modifications to input parameters to better match a recommended crop.
    """
    adjustments = {}

    # Example crop-specific thresholds (Modify as needed)
    crop_requirements = {
        "rice": {"N": 80, "P": 40, "K": 40, "temperature": 25, "humidity": 80, "ph": 6.0, "rainfall": 200},
        "maize": {"N": 60, "P": 30, "K": 30, "temperature": 20, "humidity": 60, "ph": 5.5, "rainfall": 150},
        "banana": {"N": 50, "P": 35, "K": 50, "temperature": 27, "humidity": 85, "ph": 5.2, "rainfall": 180},
    }

    best_crop = rule_suggestions[0]  # Take the first rule-based suggestion
    if best_crop in crop_requirements:
        ideal_values = crop_requirements[best_crop]

        # Compare and suggest changes
        if N < ideal_values["N"]:
            adjustments["N"] = f"Increase N from {N} to {ideal_values['N']}"
        elif N > ideal_values["N"]:
            adjustments["N"] = f"Decrease N from {N} to {ideal_values['N']}"

        if P < ideal_values["P"]:
            adjustments["P"] = f"Increase P from {P} to {ideal_values['P']}"
        elif P > ideal_values["P"]:
            adjustments["P"] = f"Decrease P from {P} to {ideal_values['P']}"

        if K < ideal_values["K"]:
            adjustments["K"] = f"Increase K from {K} to {ideal_values['K']}"
        elif K > ideal_values["K"]:
            adjustments["K"] = f"Decrease K from {K} to {ideal_values['K']}"

        if temperature < ideal_values["temperature"]:
            adjustments["temperature"] = f"Increase temperature from {temperature}°C to {ideal_values['temperature']}°C"
        elif temperature > ideal_values["temperature"]:
            adjustments["temperature"] = f"Decrease temperature from {temperature}°C to {ideal_values['temperature']}°C"

        if humidity < ideal_values["humidity"]:
            adjustments["humidity"] = f"Increase humidity from {humidity}% to {ideal_values['humidity']}%"
        elif humidity > ideal_values["humidity"]:
            adjustments["humidity"] = f"Decrease humidity from {humidity}% to {ideal_values['humidity']}%"

        if ph < ideal_values["ph"]:
            adjustments["ph"] = f"Increase pH from {ph} to {ideal_values['ph']}"
        elif ph > ideal_values["ph"]:
            adjustments["ph"] = f"Decrease pH from {ph} to {ideal_values['ph']}"

        if rainfall < ideal_values["rainfall"]:
            adjustments["rainfall"] = f"Increase rainfall from {rainfall}mm to {ideal_values['rainfall']}mm"
        elif rainfall > ideal_values["rainfall"]:
            adjustments["rainfall"] = f"Decrease rainfall from {rainfall}mm to {ideal_values['rainfall']}mm"

    return adjustments




@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        N = data['N']
        P = data['P']
        K = data['K']
        temperature = data['temperature']
        humidity = data['humidity']
        ph = data['ph']
        rainfall = data['rainfall']

        # Scale input data
        scaled_input = scaler.transform([[N, P, K, temperature, humidity, ph, rainfall]])

        # Predict using the hybrid ML model
        predicted_crop = model.predict(scaled_input)
        predicted_crop_name = label_encoder.inverse_transform(predicted_crop)[0]

        # Get rule-based suggestions
        rule_suggestions = rule_based_filter(N, P, K, temperature, humidity, ph, rainfall)

        # If ML and rule-based suggestions differ, suggest modifications
        if predicted_crop_name not in rule_suggestions:
            modifications = suggest_parameter_modifications(
                N, P, K, temperature, humidity, ph, rainfall, predicted_crop_name, rule_suggestions
            )
        else:
            modifications = {}

        return jsonify({
            "recommended_crop": {
                "ml_prediction": predicted_crop_name,
                "rule_suggestions": rule_suggestions,
                "parameter_modifications": modifications if modifications else "No major adjustments needed"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)