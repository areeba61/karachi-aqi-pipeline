# model_loader.py
import hopsworks
import joblib
import pandas as pd

def categorize_scaled_aqi(aqi):
    if aqi >= 4.5:
        return "Moderate"
    elif aqi >= 3.5:
        return "Unhealthy for Sensitive Groups"
    elif aqi >= 2.5:
        return "Unhealthy"
    elif aqi >= 1.5:
        return "Very Unhealthy"
    else:
        return "Hazardous"

print(" Logging into Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

# Load model
model = mr.get_model("karachi_aqi_forecaster", version=13)
model_dir = model.download()
rf_model = joblib.load(model_dir + "/rf_model.pkl")

# Load past AQI data
fg = fs.get_feature_group("karachi_weather_hourly", version=1)
df = fg.select_all().read().sort_values("timestamp")
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour
df["source"] = "Past"

# Build lag features
lag_values = df["aqi"].tail(72).values
input_dict = {f"aqi_lag_{i+1}": lag_values[i] for i in range(72)}
X_input = pd.DataFrame([input_dict])

# Predict next 72 hours
forecast = rf_model.predict(X_input)[0]
last_timestamp = df["timestamp"].max()
future_hours = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=72, freq='h')

# Build forecast table
forecast_df = pd.DataFrame({
    "timestamp": future_hours,
    "aqi": forecast,
    "source": "Forecast"
})
forecast_df["date"] = forecast_df["timestamp"].dt.date
forecast_df["hour"] = forecast_df["timestamp"].dt.hour

# Combine for full chart
full_df = pd.concat([df[["timestamp", "aqi", "source"]], forecast_df[["timestamp", "aqi", "source"]]])

# Daily averages + category
daily_avg = forecast_df.groupby("date")["aqi"].mean().reset_index()
daily_avg["category"] = daily_avg["aqi"].apply(categorize_scaled_aqi)
