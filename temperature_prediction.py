import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Step 1: Load CO2 emissions data
co2_data = pd.read_csv('total-ghg-emissions.csv')

# Step 2: Preprocess CO2 emissions data
co2_data['Year'] = pd.to_datetime(co2_data['Year'], format='%Y').dt.year

# Step 3: Load temperature data
temperature_data = pd.read_csv('temperature.csv')

# Step 4: Preprocess temperature data
temperature_data['year'] = pd.to_datetime(temperature_data['year'], format='%Y').dt.year
temperature_data.dropna(subset=['AverageTemperatureFahr'], inplace=True)

# Step 5: Merge CO2 emissions and temperature data
merged_data = pd.merge(co2_data, temperature_data, how='inner', left_on=['Entity', 'Year'], right_on=['Country', 'year'])

# Step 6: Model Training
# Split data into train and test sets
train_size = int(len(merged_data) * 0.8)
train, test = merged_data.iloc[:train_size], merged_data.iloc[train_size:]

# Extract the target column for prediction
target_column = 'AverageTemperatureFahr'

# Fit ARIMA model
model = ARIMA(train[target_column], exog=train['Annual greenhouse gas emissions in CO₂ equivalents'], order=(1, 1, 1))
model_fit = model.fit()

# Step 7: Prediction and Visualization
# Make predictions
forecast = model_fit.forecast(steps=len(test), exog=test['Annual greenhouse gas emissions in CO₂ equivalents'])

# Plot actual vs. predicted temperature change
plt.figure(figsize=(10, 6))
plt.plot(train.index, train[target_column], label='Training Data')
plt.plot(test.index, test[target_column], label='Test Data')
plt.plot(test.index, forecast, label='Predictions', linestyle='--')
plt.title('Temperature Change Prediction using ARIMA with CO2 Emissions')
plt.xlabel('Date')
plt.ylabel('Temperature (Fahrenheit)')
plt.legend()
plt.show()
