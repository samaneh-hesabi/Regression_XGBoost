import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_california_housing

def load_trained_model():
    # Load the California Housing dataset to get feature names
    data = fetch_california_housing()
    feature_names = data.feature_names
    
    # Create and train the model (same as in the original script)
    X = pd.DataFrame(data.data, columns=feature_names)
    y = pd.Series(data.target, name='target')
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    model.fit(X, y)
    return model, feature_names

def prepare_new_data(new_data_dict):
    """
    Prepare new data for prediction.
    new_data_dict should be a dictionary with the following keys:
    - MedInc: Median income in block group
    - HouseAge: Median house age in block group
    - AveRooms: Average number of rooms per household
    - AveBedrms: Average number of bedrooms per household
    - Population: Block group population
    - AveOccup: Average number of household members
    - Latitude: Block group latitude
    - Longitude: Block group longitude
    """
    return pd.DataFrame([new_data_dict])

def make_prediction(model, new_data):
    """
    Make prediction on new data
    """
    return model.predict(new_data)

def main():
    # Load the trained model
    model, feature_names = load_trained_model()
    
    # Example new data (you can modify these values)
    new_data_dict = {
        'MedInc': 8.3252,
        'HouseAge': 41.0,
        'AveRooms': 6.984127,
        'AveBedrms': 1.023810,
        'Population': 322.0,
        'AveOccup': 2.555556,
        'Latitude': 37.88,
        'Longitude': -122.23
    }
    
    # Prepare the new data
    new_data = prepare_new_data(new_data_dict)
    
    # Make prediction
    prediction = make_prediction(model, new_data)
    
    print("\nNew Data Features:")
    for feature, value in new_data_dict.items():
        print(f"{feature}: {value}")
    
    print(f"\nPredicted House Value: ${prediction[0] * 100000:.2f}")

if __name__ == "__main__":
    main() 