import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Load ML model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Categorical mappings
time_mapping = {
    'Early_Morning': 0, 'Morning': 1, 'Afternoon': 2, 
    'Evening': 3, 'Night': 4, 'Late_Night': 5
}

day_mapping = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

stops_mapping = {
    'zero': 0, 'one': 1, 'two_or_more': 2
}

class_mapping = {
    'Economy': 0, 'Business': 1
}

# Helper to encode categorical features
def encode_features(data):
    features = []
    
    # 1. Numerical features first (7 features)
    features.append(time_mapping[data["departure_time"]])  # departure_time
    features.append(stops_mapping[data["stops"]])         # stops
    features.append(time_mapping[data["arrival_time"]])    # arrival_time
    features.append(class_mapping[data["class"]])         # class
    features.append(float(data["duration"]))              # duration
    features.append(int(data["days_left"]))               # days_left
    features.append(day_mapping[data["day"]])             # day
    
    # 2. One-hot encode airline (6 features)
    airlines = ["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"]
    features += [int(data["airline"] == a) for a in airlines]
    
    # 3. One-hot encode source_city (13 features)
    source_cities = ["Ahmedabad", "Bangalore", "Chandigarh", "Chennai", "Coimbatore", 
                    "Delhi", "Hyderabad", "Jaipur", "Kolkata", "Lucknow", "Mumbai", "Pune"]
    features += [int(data["source_city"] == c) for c in source_cities]
    
    # 4. One-hot encode destination_city (13 features)
    dest_cities = ["Ahmedabad", "Bangalore", "Chandigarh", "Chennai", "Coimbatore", 
                  "Delhi", "Hyderabad", "Jaipur", "Kolkata", "Lucknow", "Mumbai", "Pune"]
    features += [int(data["destination_city"] == c) for c in dest_cities]
    
    return np.array(features).reshape(1, -1)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Flight Fare Prediction API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        # First encode all features
        X = encode_features(data)
        
        # Scale only the numerical features (days_left and duration)
        # These are the last 2 features in our encoded array
        numerical_features = X[:, -2:]  # Last 2 columns
        scaled_numerical = scaler.transform(numerical_features)
        
        # Replace the numerical features with scaled ones
        X_scaled = X.copy()
        X_scaled[:, -2:] = scaled_numerical
        
        pred = model.predict(X_scaled)
        return jsonify({"fare": float(pred[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
