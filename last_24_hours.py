import requests, pandas as pd, numpy as np, time, os
from datetime import datetime, timedelta
import hopsworks, joblib
from hsml.schema import Schema
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ========== CONFIG ==========
API_KEY = "1d17e1a5e315042bcdc770c37af4ed0d"
LAT, LON = 24.8607, 67.0011
CITY_ID = 1174872
CSV_PATH = "karachi_weather_hourly.csv"

# ========== FETCH FUNCTIONS ==========
def fetch_weather_at(ts):
    url = "http://history.openweathermap.org/data/2.5/history/city"
    params = {
        "id": CITY_ID, "type": "hour",
        "start": int(ts.timestamp()), "end": int((ts + timedelta(hours=1)).timestamp()),
        "appid": API_KEY
    }
    r = requests.get(url, params=params)
    return r.json()["list"][0] if r.status_code == 200 and r.json().get("list") else None

def fetch_pollution_at(ts):
    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": LAT, "lon": LON,
        "start": int(ts.timestamp()), "end": int((ts + timedelta(hours=1)).timestamp()),
        "appid": API_KEY
    }
    r = requests.get(url, params=params)
    return r.json()["list"][0] if r.status_code == 200 and r.json().get("list") else None

def build_record(ts):
    weather = fetch_weather_at(ts)
    pollution = fetch_pollution_at(ts)
    time.sleep(1)
    if not weather or not pollution: return None
    comp = pollution["components"]
    return {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "aqi": pollution["main"]["aqi"],
        "pm2_5": comp["pm2_5"], "pm10": comp["pm10"],
        "temperature": round(weather["main"]["temp"] - 273.15, 2),
        "humidity": weather["main"]["humidity"],
        "wind_speed": weather["wind"]["speed"]
    }

def collect_hourly_data(start, end):
    records = []
    while start < end:
        print(f" Fetching {start}")
        rec = build_record(start)
        if rec: records.append(rec)
        start += timedelta(hours=1)
    df = pd.DataFrame(records)
    if os.path.exists(CSV_PATH):
        old = pd.read_csv(CSV_PATH)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=["timestamp"])
    df.to_csv(CSV_PATH, index=False)
    print(f" Saved {len(df)} total records to {CSV_PATH}")

# ========== MAIN ==========
if __name__ == "__main__":
    #  Fetch last 24h data
    start = datetime.now() - timedelta(days=1)
    end = datetime.now()
    collect_hourly_data(start, end)

    #  Insert into Feature Store
    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fg = fs.get_or_create_feature_group(
        name="karachi_weather_hourly", version=1,
        description="Hourly weather and pollution data for Karachi",
        primary_key=["timestamp"], event_time="timestamp"
    )
    fg.insert(df)
    print(" Last 24h data inserted into Feature Store")

    #  Read full dataset
    df = fg.select_all().read().sort_values("timestamp").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df.drop(columns=["timestamp"], inplace=True)

    #  Outlier capping
    def cap_outliers(df, cols):
        for col in cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower, upper)
        return df

    df = cap_outliers(df, ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "aqi"])

    #  Lag features
    def prepare_multioutput_forecast_data(df, lag=72, horizon=72):
        if len(df) < lag + horizon: return None, None
        lag_df = pd.concat([df["aqi"].shift(i) for i in range(1, lag + 1)], axis=1)
        lag_df.columns = [f"aqi_lag_{i}" for i in range(1, lag + 1)]
        target_df = pd.concat([df["aqi"].shift(-i) for i in range(1, horizon + 1)], axis=1)
        target_df.columns = [f"target_t_plus_{i}" for i in range(1, horizon + 1)]
        final_df = pd.concat([lag_df, target_df], axis=1).dropna()
        return final_df[lag_df.columns], final_df[target_df.columns]

    X, y = prepare_multioutput_forecast_data(df)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    #  Train model
    rf_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    rf_model.fit(X_train, y_train)

    #  Evaluate
    test_pred = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, test_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    rf_r2 = r2_score(y_test, test_pred)
    rf_acc = max(0, 1 - (rf_mae / np.mean(y_test)))

    print(" Testing Performance")
    print("MAE:", round(rf_mae, 2), "RMSE:", round(rf_rmse, 2),
          "R²:", round(rf_r2, 4), "Accuracy:", round(rf_acc * 100, 2), "%")

    #  Save model
    joblib.dump(rf_model, "rf_model.pkl", compress=3)
    input_example = X.iloc[0]
    model_schema = Schema(X)

    model = mr.python.create_model(
        name="karachi_aqi_forecaster",
        metrics={
            "test_accuracy": round(rf_acc, 4),
            "test_mae": round(rf_mae, 2),
            "test_rmse": round(rf_rmse, 2),
            "test_r2": round(rf_r2, 4)
        },
        model_schema=model_schema,
        input_example=input_example,
        description="Random Forest retrained daily with last 24h data"
    )
    model.save("rf_model.pkl")
    print(f" Model saved to Hopsworks with TEST accuracy: {round(rf_acc * 100, 2)}%")

