import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import datetime  # optional for logging pings

app = Flask(__name__)
CORS(
    app,
    origins=["https://flight-fare-prediction-coral.vercel.app"],
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"]
)

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
    features.append(time_mapping[data["departure_time"]])
    features.append(stops_mapping[data["stops"]])
    features.append(time_mapping[data["arrival_time"]])
    features.append(class_mapping[data["class"]])
    features.append(float(data["duration"]))
    features.append(int(data["days_left"]))
    features.append(day_mapping[data["day"]])
    
    # 2. One-hot encode airline (6 features)
    airlines = ["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"]
    features += [int(data["airline"] == a) for a in airlines]
    
    # 3. One-hot encode source_city (12 features)
    source_cities = ["Ahmedabad", "Bangalore", "Chandigarh", "Chennai", "Coimbatore", 
                    "Delhi", "Hyderabad", "Jaipur", "Kolkata", "Lucknow", "Mumbai", "Pune"]
    features += [int(data["source_city"] == c) for c in source_cities]
    
    # 4. One-hot encode destination_city (12 features)
    dest_cities = ["Ahmedabad", "Bangalore", "Chandigarh", "Chennai", "Coimbatore", 
                  "Delhi", "Hyderabad", "Jaipur", "Kolkata", "Lucknow", "Mumbai", "Pune"]
    features += [int(data["destination_city"] == c) for c in dest_cities]
    
    return np.array(features).reshape(1, -1)

# ✅ Health check route for browser/manual checking
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Flight Fare Prediction API is running"})

# ✅ Cron-job-friendly ping route (very lightweight)
@app.route("/ping", methods=["GET"])
def ping():
    print("Ping received at", datetime.datetime.now())
    return "pong", 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        X = encode_features(data)
        numerical_features = X[:, -2:]
        scaled_numerical = scaler.transform(numerical_features)
        X_scaled = X.copy()
        X_scaled[:, -2:] = scaled_numerical
        pred = model.predict(X_scaled)
        return jsonify({"fare": float(pred[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
