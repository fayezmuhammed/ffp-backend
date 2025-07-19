import pandas as pd
import numpy as np
import joblib

# Load the model to check its feature names
model = joblib.load("model.pkl")
print("Model feature names:", model.feature_names_in_)
print("Number of features:", len(model.feature_names_in_))

# Load some sample data to see the structure
data = pd.read_csv("expanded_flights.csv")
print("\nSample data columns:", data.columns.tolist())

# Check the one-hot encoded structure
columns_to_encode = ['airline', 'source_city', 'destination_city']
data_encoded = pd.get_dummies(data, columns=columns_to_encode, dtype=int)
print("\nEncoded data columns:", data_encoded.columns.tolist())
print("Number of encoded columns:", len(data_encoded.columns)) 