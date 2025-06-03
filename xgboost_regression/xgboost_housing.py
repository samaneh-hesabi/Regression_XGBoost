# Import required libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("California Housing Price Prediction using XGBoost")
print("=" * 50)

# Step 1: Load and prepare the data
print("\nStep 1: Loading and preparing the data...")
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Print information about the features
print("\nFeatures in the dataset:")
for feature in data.feature_names:
    print(f"- {feature}")
print(f"\nTotal number of samples: {len(X)}")

# Step 2: Split the data
print("\nStep 2: Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Step 3: Create and train the model
print("\nStep 3: Training the XGBoost model...")
model = xgb.XGBRegressor(
    objective='reg:squarederror',  # For regression problems
    n_estimators=100,              # Number of trees
    learning_rate=0.1,             # Learning rate
    max_depth=6,                   # Maximum tree depth
    random_state=42                # For reproducibility
)
model.fit(X_train, y_train)
print("Model training completed!")

# Step 4: Make predictions and evaluate
print("\nStep 4: Evaluating the model...")
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Step 5: Analyze feature importance
print("\nStep 5: Analyzing feature importance...")
feature_importance = pd.DataFrame({
    'Feature': data.feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance Ranking:")
for idx, row in feature_importance.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")

# Step 6: Create visualizations
print("\nStep 6: Creating visualizations...")
plt.figure(figsize=(15, 6))

# Plot 1: Actual vs Predicted Values
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual House Values')
plt.ylabel('Predicted House Values')
plt.title('Actual vs Predicted House Values')

# Plot 2: Feature Importance
plt.subplot(1, 2, 2)
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in Price Prediction')

plt.tight_layout()
plt.savefig('results/housing_prediction_results.png')
print("Visualizations saved as 'results/housing_prediction_results.png'")
plt.close()

# Step 7: Example prediction for a new house
print("\nExample: Predicting price for a new house...")
example_house = {
    'MedInc': 8.3252,      # Median income in block
    'HouseAge': 41.0,      # House age
    'AveRooms': 6.984127,  # Average rooms
    'AveBedrms': 1.023810, # Average bedrooms
    'Population': 322.0,    # Population in block
    'AveOccup': 2.555556,  # Average occupancy
    'Latitude': 37.88,     # Latitude
    'Longitude': -122.23   # Longitude
}

# Make prediction for the example house
new_data = pd.DataFrame([example_house])
prediction = model.predict(new_data)[0]
predicted_price = prediction * 100000  # Convert to actual price (target is in $100,000 units)

print(f"\nPredicted house price: ${predicted_price:,.2f}") 