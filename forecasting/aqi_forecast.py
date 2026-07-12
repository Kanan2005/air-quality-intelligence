"""
Hyperlocal AQI forecasting.

Approach (deliberately kept simple + explainable for a hackathon demo,
but genuinely trained on real station history, not hardcoded):

  - Feature engineer hour-of-day, day-of-week, and lag features (t-1h,
    t-24h, t-168h) per station.
  - Train a RandomForestRegressor per city (fast, no GPU, handles
    non-linear diurnal patterns well, and gives feature_importances_
    for the "why" behind a forecast -- useful for judges' Q&A).
  - Forecast is generated recursively hour-by-hour up to `horizon_hours`.

This benchmarks itself against a "persistence baseline" (tomorrow = today),
which is exactly the baseline the hackathon's own evaluation criteria
references ("AQI forecast accuracy... versus persistence baseline") --
worth keeping in your demo/deck.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def _add_lag_features(df: pd.DataFrame, station_col="station", value_col="aqi") -> pd.DataFrame:
    df = df.sort_values([station_col, "timestamp"]).copy()
    g = df.groupby(station_col)[value_col]
    df["lag_1h"] = g.shift(1)
    df["lag_24h"] = g.shift(24)
    df["lag_168h"] = g.shift(168)
    df["roll_24h_mean"] = g.transform(lambda s: s.shift(1).rolling(24, min_periods=6).mean())
    return df


FEATURE_COLS = ["hour", "dow", "is_weekend", "lag_1h", "lag_24h", "lag_168h", "roll_24h_mean"]


class AQIForecaster:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        self.trained = False
        self.mae_ = None
        self.baseline_mae_ = None

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _add_time_features(df)
        df = _add_lag_features(df)
        return df.dropna(subset=FEATURE_COLS + ["aqi"])

    def fit(self, df: pd.DataFrame, holdout_hours: int = 48):
        """Trains on all-but-last `holdout_hours`, evaluates on the holdout,
        and reports MAE vs persistence baseline (t-24h) for that holdout."""
        prepped = self.prepare(df)
        cutoff = prepped["timestamp"].max() - pd.Timedelta(hours=holdout_hours)
        train = prepped[prepped["timestamp"] <= cutoff]
        test = prepped[prepped["timestamp"] > cutoff]

        self.model.fit(train[FEATURE_COLS], train["aqi"])
        self.trained = True

        if len(test):
            preds = self.model.predict(test[FEATURE_COLS])
            self.mae_ = mean_absolute_error(test["aqi"], preds)
            self.baseline_mae_ = mean_absolute_error(test["aqi"], test["lag_24h"])
        return self

    def feature_importances(self):
        if not self.trained:
            return {}
        return dict(sorted(
            zip(FEATURE_COLS, self.model.feature_importances_),
            key=lambda x: -x[1]
        ))

    def forecast_station(self, history: pd.DataFrame, horizon_hours: int = 72) -> pd.DataFrame:
        """
        Recursive multi-step forecast for a single station's history
        (must contain columns: timestamp, aqi, sorted ascending).
        """
        if not self.trained:
            raise RuntimeError("Call .fit() before forecasting.")

        hist = history[["timestamp", "aqi"]].sort_values("timestamp").copy()
        last_ts = hist["timestamp"].max()
        preds = []

        for step in range(1, horizon_hours + 1):
            future_ts = last_ts + pd.Timedelta(hours=step)
            lag_1h = hist["aqi"].iloc[-1]
            lag_24h = hist.set_index("timestamp")["aqi"].reindex(
                [future_ts - pd.Timedelta(hours=24)], method="nearest"
            ).iloc[0]
            lag_168h = hist.set_index("timestamp")["aqi"].reindex(
                [future_ts - pd.Timedelta(hours=168)], method="nearest"
            ).iloc[0]
            roll_24h = hist["aqi"].tail(24).mean()

            feat = pd.DataFrame([{
                "hour": future_ts.hour,
                "dow": future_ts.dayofweek,
                "is_weekend": int(future_ts.dayofweek >= 5),
                "lag_1h": lag_1h,
                "lag_24h": lag_24h,
                "lag_168h": lag_168h,
                "roll_24h_mean": roll_24h,
            }])
            pred = float(self.model.predict(feat[FEATURE_COLS])[0])
            preds.append({"timestamp": future_ts, "predicted_aqi": round(pred, 1)})

            hist = pd.concat([hist, pd.DataFrame([{"timestamp": future_ts, "aqi": pred}])],
                              ignore_index=True)

        return pd.DataFrame(preds)


if __name__ == "__main__":
    df = pd.read_csv("../data/sample_aqi_data.csv", parse_dates=["timestamp"])
    delhi = df[df["city"] == "Delhi"]

    fc = AQIForecaster().fit(delhi)
    print(f"Model MAE: {fc.mae_:.1f} | Persistence baseline MAE: {fc.baseline_mae_:.1f}")
    print("Feature importances:", fc.feature_importances())

    station = delhi[delhi["station"] == delhi["station"].iloc[0]]
    forecast = fc.forecast_station(station, horizon_hours=24)
    print(forecast.head(10))
