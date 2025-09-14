# %% [code] {"jupyter":{"outputs_hidden":false}}
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import random
import datetime
import os
import joblib
import pytz

# %% [code] {"jupyter":{"outputs_hidden":false}}
data=pd.read_csv("/kaggle/input/solar-thing/accurate_solar_power_dataset.csv")
data.head()
data=data.drop(columns=['weather_condition','is_raining'])

# %% [code] {"jupyter":{"outputs_hidden":false}}
data.isnull().sum()

# %% [code] {"jupyter":{"outputs_hidden":false}}
X=data[[	'hour','day_of_year',	'month', 'latitude',	'longitude',	'panel_surface_area_m2',	'panel_tilt_degrees',	'panel_azimuth_degrees',	'panel_efficiency', 'solar_irradiance_w_m2',	'solar_elevation_degrees',	'temperature_celsius',	'humidity_percent',	'wind_speed_ms',	'cloud_cover_percent',	'seasonal_factor']]
y=data['power_output_kw']

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [code] {"jupyter":{"outputs_hidden":false}}
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false}}
y_pred = rf.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# %% [code] {"jupyter":{"outputs_hidden":false}}
# set seed for reproducibility
np.random.seed(42)

# define ranges
AREA_RANGE = (1.0, 10.0)   # m²
EFF_RANGE  = (0.12, 0.22)  # efficiency fraction
TILT_RANGE = (0.0, 45.0)   # degrees
AZ_RANGE   = (90.0, 270.0) # degrees

# create synthetic panel features
data['panel_surface_area_m2'] = np.random.uniform(*AREA_RANGE, size=len(data))
data['panel_efficiency']     = np.random.uniform(*EFF_RANGE, size=len(data))
data['panel_tilt_degrees']   = np.random.uniform(*TILT_RANGE, size=len(data))
data['panel_azimuth_degrees'] = np.random.uniform(*AZ_RANGE, size=len(data))

# quick check
data[['panel_surface_area_m2','panel_efficiency','panel_tilt_degrees','panel_azimuth_degrees']].head()

# %% [code] {"execution":{"iopub.status.busy":"2025-09-14T01:38:12.894188Z","iopub.execute_input":"2025-09-14T01:38:12.895024Z","iopub.status.idle":"2025-09-14T01:39:09.424679Z","shell.execute_reply.started":"2025-09-14T01:38:12.894994Z","shell.execute_reply":"2025-09-14T01:39:09.423553Z"},"jupyter":{"outputs_hidden":false}}
import datetime
import random
import pytz
import math


city_to_coords = {
    "thiruvananthapuram": (8.5, 76.9),
    "mumbai": (19.0, 72.8),
    "gandhinagar": (23.2, 72.6),
    "panaji": (15.5, 73.8),
    "bengaluru": (12.9, 77.6),
    "chennai": (13.1, 80.3),
    "hyderabad": (17.4, 78.5),
    "amaravati": (16.5, 80.5),
    "bhubaneswar": (20.3, 85.8),
    "kolkata": (22.6, 88.4),
    "ranchi": (23.3, 85.3),
    "patna": (25.6, 85.1),
    "lucknow": (26.8, 80.9),
    "dehradun": (30.3, 78.0),
    "chandigarh": (30.7, 76.8),
    "shimla": (31.1, 77.2),
    "srinagar": (34.1, 74.8),
    "jammu": (32.7, 74.9),
    "raipur": (21.3, 81.6),
    "bhopal": (23.3, 77.4),
    "jaipur": (26.9, 75.8),
    "dispur": (26.1, 91.8),
    "agartala": (23.8, 91.3),
    "aizawl": (23.7, 92.7),
    "shillong": (25.6, 91.9),
    "imphal": (24.8, 93.9),
    "kohima": (25.7, 94.1),
    "itanagar": (27.1, 93.6),
    "gangtok": (27.3, 88.6),
    "delhi": (28.6, 77.2),
    "puducherry": (11.9, 79.8),
    "port_blair": (11.7, 92.7),
    "kavaratti": (10.6, 72.6),
    "daman": (20.4, 72.8),
    "silvassa": (20.3, 73.0),
    "leh": (34.2, 77.6)
}

# ADD THIS FUNCTION - the one that uses your trained model
def predict_power(hour, day_of_year, month, latitude, longitude,
                  panel_surface_area_m2, panel_tilt_degrees, panel_azimuth_degrees,
                  panel_efficiency, solar_irradiance_w_m2, solar_elevation_degrees,
                  temperature_celsius, humidity_percent, wind_speed_ms,
                  cloud_cover_percent, seasonal_factor):
    
    features = [[hour, day_of_year, month, latitude, longitude,
                 panel_surface_area_m2, panel_tilt_degrees, panel_azimuth_degrees,
                 panel_efficiency, solar_irradiance_w_m2, solar_elevation_degrees,
                 temperature_celsius, humidity_percent, wind_speed_ms,
                 cloud_cover_percent, seasonal_factor]]
    
    return rf.predict(features)[0]

def prepare_inputs(city, panel_surface_area_m2, panel_tilt_degrees,
                   panel_azimuth_degrees, panel_efficiency, hour):
    # --- get datetime features
    now = datetime.datetime.now()

    # For India Standard Time
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.datetime.now(ist)
    hour = now.hour
    day_of_year = now.timetuple().tm_yday
    month = now.month
    
    # --- get lat/lon from city (with error handling)
    city_lower = city.lower().strip()
    if city_lower not in city_to_coords:
        available_cities = ", ".join(city_to_coords.keys())
        raise ValueError(f"City '{city}' not found. Available cities: {available_cities}")
    
    latitude, longitude = city_to_coords[city_lower]
    
    # --- mock values for other features
    mock_inputs = {
        "hour": hour,
        "day_of_year": day_of_year,
        "month": month,
        "latitude": latitude,
        "longitude": longitude,
        "panel_surface_area_m2": panel_surface_area_m2,
        "panel_tilt_degrees": panel_tilt_degrees,
        "panel_azimuth_degrees": panel_azimuth_degrees,
        "panel_efficiency": panel_efficiency,
        "solar_irradiance_w_m2": random.uniform(600, 1000),
        "solar_elevation_degrees": random.uniform(30, 70),
        "temperature_celsius": random.uniform(20, 40),
        "humidity_percent": random.uniform(30, 70),
        "wind_speed_ms": random.uniform(0, 5),
        "cloud_cover_percent": random.uniform(0, 50),
        "seasonal_factor": 1.0
    }
    return mock_inputs

# Get user inputs
print("Solar Power Generation Predictor")
print("=" * 40)

# Display available cities
print("\nAvailable cities:")
cities = list(city_to_coords.keys())
for i, city_name in enumerate(cities, 1):
    print(f"{i}. {city_name.title()}")

print("\nEnter your solar panel configuration:")

# Get city input
while True:
    city_input = input("City name (from the list above): ").lower().strip()
    if city_input in city_to_coords:
        city = city_input
        break
    else:
        print("Invalid city. Please choose from the list above.")

# Get panel specifications
while True:
    try:
        panel_area = float(input("Panel Area (m²): "))
        if panel_area > 0:
            break
        else:
            print("Panel area must be positive.")
    except ValueError:
        print("Please enter a valid number.")

while True:
    try:
        tilt = float(input("Panel Tilt (degrees, 0-90): "))
        if 0 <= tilt <= 90:
            break
        else:
            print("Tilt must be between 0 and 90 degrees.")
    except ValueError:
        print("Please enter a valid number.")

while True:
    try:
        azimuth = float(input("Azimuth Angle (degrees, usually 180 for south-facing): "))
        if 0 <= azimuth <= 360:
            break
        else:
            print("Azimuth must be between 0 and 360 degrees.")
    except ValueError:
        print("Please enter a valid number.")

while True:
    try:
        efficiency = float(input("Panel Efficiency (0.15-0.25, e.g., 0.20 for 20%): "))
        if 0.1 <= efficiency <= 1.0:
            break
        else:
            print("Efficiency must be between 0.1 and 1.0")
    except ValueError:
        print("Please enter a valid number.")

print(f"\nTesting solar panel configuration:")
print(f"City: {city.title()}")
print(f"Panel Area: {panel_area} m²")
print(f"Tilt: {tilt}°")
print(f"Azimuth: {azimuth}°")
print(f"Efficiency: {efficiency}")
print("-" * 40)

try:
    # Build all required features
    inputs = prepare_inputs(city, panel_area, tilt, azimuth, efficiency, hour)
    
    # --- Get idealized prediction first (before corrections)
    idealized_pred = predict_power(
        inputs['hour'], 
        inputs['day_of_year'], 
        inputs['month'],
        inputs['latitude'], 
        inputs['longitude'],
        inputs['panel_surface_area_m2'], 
        inputs['panel_tilt_degrees'], 
        inputs['panel_azimuth_degrees'],
        inputs['panel_efficiency'],
        inputs['solar_irradiance_w_m2'], 
        inputs['solar_elevation_degrees'],
        inputs['temperature_celsius'], 
        inputs['humidity_percent'], 
        inputs['wind_speed_ms'],
        inputs['cloud_cover_percent'], 
        inputs['seasonal_factor']
    )
    
    # --- Apply real-world correction factors
    # Ensure night handling first
    if inputs['hour'] < 6 or inputs['hour'] > 18:
        inputs["solar_irradiance_w_m2"] = 0
        inputs["solar_elevation_degrees"] = 0
    
    # Calculate correction factors
    angle_factor = max(0, math.cos(math.radians(abs(inputs['panel_tilt_degrees'] - inputs['solar_elevation_degrees']))))
    temp_factor = 1 - 0.005 * max(0, inputs['temperature_celsius'] - 25)
    cloud_factor = 1 - (inputs['cloud_cover_percent'] / 100)
    system_derate = 0.8
    
    # Apply all correction factors to irradiance
    original_irradiance = inputs["solar_irradiance_w_m2"]
    inputs["solar_irradiance_w_m2"] *= angle_factor * temp_factor * cloud_factor * system_derate
    
    # --- Get realistic prediction with corrected inputs
    realistic_pred = predict_power(
        inputs['hour'], 
        inputs['day_of_year'], 
        inputs['month'],
        inputs['latitude'], 
        inputs['longitude'],
        inputs['panel_surface_area_m2'], 
        inputs['panel_tilt_degrees'], 
        inputs['panel_azimuth_degrees'],
        inputs['panel_efficiency'],
        inputs['solar_irradiance_w_m2'], 
        inputs['solar_elevation_degrees'],
        inputs['temperature_celsius'], 
        inputs['humidity_percent'], 
        inputs['wind_speed_ms'],
        inputs['cloud_cover_percent'], 
        inputs['seasonal_factor']
    )
    
    # Convert to kW if needed
    if idealized_pred > 1000:
        idealized_pred_kw = idealized_pred / 1000.0
    else:
        idealized_pred_kw = float(idealized_pred)
        
    if realistic_pred > 1000:
        realistic_pred_kw = realistic_pred / 1000.0
    else:
        realistic_pred_kw = float(realistic_pred)
    
    # Ensure no negative values
    idealized_pred_kw = max(0.0, idealized_pred_kw)
    realistic_pred_kw = max(0.0, realistic_pred_kw)
    
    # Print correction factor details
    print(f"\n--- Real-World Correction Factors ---")
    print(f"Original irradiance: {original_irradiance:.1f} W/m²")
    print(f"Angle factor (tilt vs elevation): {angle_factor:.3f}")
    print(f"Temperature factor (derating): {temp_factor:.3f}")
    print(f"Cloud cover factor: {cloud_factor:.3f}")
    print(f"System derating factor: {system_derate:.3f}")
    print(f"Corrected irradiance: {inputs['solar_irradiance_w_m2']:.1f} W/m²")
    
    # Print both predictions
    print(f"\n--- SOLAR POWER PREDICTIONS ---")
    print(f"Idealized prediction (no corrections): {idealized_pred_kw:.3f} kW")
    print(f"Realistic corrected prediction: {realistic_pred_kw:.3f} kW")
    print(f"Reality factor: {realistic_pred_kw/max(0.001, idealized_pred_kw):.2f}x")
    
    predicted = realistic_pred_kw
    
    # Print some of the input features for verification
    print("\nFinal input features used:")
    for key, value in inputs.items():
        print(f"{key}: {value}")
        
except Exception as e:
    print(f"Error: {e}")
