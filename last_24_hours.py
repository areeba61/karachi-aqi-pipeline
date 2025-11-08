import requests, pandas as pd, numpy as np, time, os, sys
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
    r = requests.get(url, params=params, timeout=30)
    return r.json()["list"][0] if r.status_code == 200 and r.json().get("list") else None

def fetch_pollution_at(ts):
    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": LAT, "lon": LON,
        "start": int(ts.timestamp()), "end": int((ts + timedelta(hours=1)).timestamp()),
        "appid": API_KEY
    }
    r = requests.get(url, params=params, timeout=30)
    return r.json()["list"][0] if r.status_code == 200 and r.json().get("list") else None

def build_record(ts):
    weather = fetch_weather_at(ts)
    pollution = fetch_pollution_at(ts)
    time.sleep(1)  # be kind to APIs
    if not weather or not pollution:
        print(f" Skipped {ts}: incomplete data")
        return None
    comp = pollution["components"]
    return {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "aqi": pollution["main"]["aqi"],
        "pm2_5": comp.get("pm2_5"),
        "pm10": comp.get("pm10"),
        "temperature": round(weather["main"]["temp"] - 273.15, 2),
        "humidity": weather["main"]["humidity"],
        "wind_speed": weather["wind"]["speed"]
    }

def collect_hourly_data(start, end):
    records = []
    while start < end:
        print(f" Fetching {start}")
        rec = build_record(start)
        if rec:
            records.append(rec)
        start += timedelta(hours=1)
    df_new = pd.DataFrame(records)
    if os.path.exists(CSV_PATH):
        old = pd.read_csv(CSV_PATH)
        df_all = pd.concat([old, df_new], ignore_index=True).drop_duplicates(subset=["timestamp"])
    else:
        df_all = df_new
    df_all.to_csv(CSV_PATH, index=False)
    print(f" Saved {len(df_all)} total records to {CSV_PATH}")

# ========== FEATURE ENGINEERING ==========
def cap_outliers(df, cols):
    df = df.copy()
    for col in cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)
    return df

def prepare_multioutput_forecast_data(df, lag=72, horizon=72):
    if len(df) < lag + horizon:
        print(f" Not enough rows. Need at least {lag + horizon}, got {len(df)}.")
        return None, None
    lag_df = pd.concat([df["aqi"].shift(i) for i in range(1, lag + 1)], axis=1)
    lag_df.columns = [f"aqi_lag_{i}" for i in range(1, lag + 1)]
    target_df = pd.concat([df["aqi"].shift(-i) for i in range(1, horizon + 1)], axis=1)
    target_df.columns = [f"target_t_plus_{i}" for i in range(1, horizon + 1)]
    final_df = pd.concat([lag_df, target_df], axis=1).dropna()
    return final_df[lag_df.columns], final_df[target_df.columns]

# ========== MAIN ==========
if __name__ == "__main__":
    # 1) Fetch last 24h
    start = datetime.now() - timedelta(days=1)
    end = datetime.now()
    collect_hourly_data(start, end)

    # 2) Insert into Feature Store
    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fg = fs.get_or_create_feature_group(
        name="karachi_weather_hourly",
        version=1,
        description="Hourly weather and pollution data for Karachi",
        primary_key=["timestamp"],
        event_time="timestamp"
    )
    fg.insert(df)
    print("Last 24h data inserted into Feature Store")

    # 3) Read merged dataset
    df = fg.select_all().read().sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        print(" No data in Feature Store, aborting.")
        sys.exit(1)

    # 4) Preprocess
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df.drop(columns=["timestamp"], inplace=True)
    df = cap_outliers(df, ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "aqi"])

    # 5) Features/targets
    X, y = prepare_multioutput_forecast_data(df)
    if X is None or y is None:
        print(" Feature generation failed (insufficient data).")
        sys.exit(1)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    # 6) Train
    rf_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    rf_model.fit(X_train, y_train)

    # 7) Evaluate on TEST
    test_pred = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, test_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    rf_r2 = r2_score(y_test, test_pred)
    rf_acc = max(0, 1 - (rf_mae / np.mean(y_test)))

    print(" Testing Performance")
    print("MAE:", round(rf_mae, 2), "RMSE:", round(rf_rmse, 2),
          "R²:", round(rf_r2, 4), "Accuracy:", round(rf_acc * 100, 2), "%")

    # 8) Save model + metrics
    joblib.dump(rf_model, "rf_model.pkl", compress=3)
    input_example = X.iloc[0]
    model_schema = Schema(X)

    model = mr.python.create_model(
        name="karachi_aqi_forecaster",
        metrics={
            "accuracy": round(rf_acc, 4),         # put in 'accuracy' so it shows in the table
            "test_accuracy": round(rf_acc, 4),
            "test_mae": round(rf_mae, 2),
            "test_rmse": round(rf_rmse, 2),
            "test_r2": round(rf_r2, 4)
        },
        model_schema=model_schema,
        input_example=input_example,
        description="Random Forest retrained daily with last 24h data"
    )
    saved_model = model.save("rf_model.pkl")
    # Print the new version if available
    try:
        print(f" Model saved. Version: {saved_model.version if hasattr(saved_model, 'version') else 'N/A'}")
    except Exception as e:
        print(f" Model saved but version fetch failed: {e}")
    print(f" Metrics saved with accuracy: {round(rf_acc * 100, 2)}%")

