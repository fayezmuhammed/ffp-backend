# Flight Fare Prediction Backend

This is the Flask backend for the Flight Fare Prediction application.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Model Files Setup

**Important**: The model files (`model.pkl` and `scaler.pkl`) are too large for GitHub and are not included in this repository.

You have two options:

#### Option A: Download Pre-trained Models
Download the model files from a cloud storage service (you'll need to upload them first):
- `model.pkl` (~706 MB)
- `scaler.pkl` (~1 KB)

#### Option B: Train Your Own Model
1. Run the training script:
```bash
python final_project_ict.py
```

2. This will generate:
- `model.pkl` - The trained Random Forest model
- `scaler.pkl` - The fitted StandardScaler

### 3. Run the Application
```bash
python app.py
```

The API will be available at `http://127.0.0.1:5000`

## API Endpoints

- `GET /` - Health check
- `POST /predict` - Predict flight fare

### Example Request
```json
{
  "airline": "AirAsia",
  "source_city": "Bangalore",
  "destination_city": "Delhi",
  "departure_time": "Morning",
  "arrival_time": "Evening",
  "day": "Monday",
  "stops": "zero",
  "class": "Economy",
  "duration": "2.5",
  "days_left": "30"
}
```

### Example Response
```json
{
  "fare": 4320.18
}
```

## Deployment

See the main `DEPLOYMENT.md` file for deployment instructions.

## Files Structure
```
backend/
├── app.py              # Flask application
├── requirements.txt    # Python dependencies
├── model.pkl          # ML model (not in repo - too large)
├── scaler.pkl         # Scaler (not in repo - too large)
├── Dockerfile         # Docker configuration
└── README.md          # This file
```
