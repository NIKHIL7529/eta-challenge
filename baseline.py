#!/usr/bin/env python
"""Baseline: gradient-boosted trees on simple engineered features.

Trains in ~5 minutes on a laptop CPU. Produces `model.pkl` which `predict.py`
loads at inference.

Prerequisites:
    python data/download_data.py   # one-time, ~500 MB download

Run:
    python baseline.py             # trains and saves model.pkl

Your job is to replace this file with something better. The grader only cares
about `predict.py` — this file just needs to produce a `model.pkl` that
`predict.py` can load.
"""

from __future__ import annotations

import sys
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"
SKIP_TRAIN = "--skip-train" in sys.argv

FEATURES = ["pickup_zone", "dropoff_zone", "hour", "dow", "month", "passenger_count", "distance", "zone_pair_mean", "zone_time_mean"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw request columns into model features."""
    ts = pd.to_datetime(df["requested_at"])
    # load coords once
    coords = pickle.load(open(Path(__file__).parent / "zone_coords.pkl", "rb"))

    def compute_distance(row):
        lat1, lon1 = coords.get(row["pickup_zone"], (0.0, 0.0))
        lat2, lon2 = coords.get(row["dropoff_zone"], (0.0, 0.0))

        # inline haversine (avoid import issues)
        import math
        R = 6371
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    df_out = pd.DataFrame({
        "pickup_zone":     df["pickup_zone"].astype("int32"),
        "dropoff_zone":    df["dropoff_zone"].astype("int32"),
        "hour":            ts.dt.hour.astype("int8"),
        "dow":             ts.dt.dayofweek.astype("int8"),
        "month":           ts.dt.month.astype("int8"),
        "passenger_count": df["passenger_count"].astype("int8"),
        "distance":        df.apply(compute_distance, axis=1).astype("float32"),
    })

    # Zone lookup maps must be passed in or computed before calling this
    df_out["zone_pair_mean"] = df_out.apply(
        lambda r: _ZONE_PAIR_MAP.get((int(r["pickup_zone"]), int(r["dropoff_zone"])), _GLOBAL_MEAN),
        axis=1
    ).astype("float32")
    df_out["zone_time_mean"] = df_out.apply(
        lambda r: _ZONE_TIME_MAP.get((int(r["pickup_zone"]), int(r["dropoff_zone"]), int(r["hour"])), r["zone_pair_mean"]),
        axis=1
    ).astype("float32")

    return df_out[FEATURES]

def main() -> None:
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"
    for p in (train_path, dev_path):
        if not p.exists():
            raise SystemExit(
                f"Missing {p.name}. Run `python data/download_data.py` first."
            )

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    print(f"  train: {len(train):,} rows")
    print(f"  dev:   {len(dev):,} rows")

    # Build zone maps from training data only
    print("Building zone lookup maps...")
    train_ts = pd.to_datetime(train["requested_at"])
    train["_hour"] = train_ts.dt.hour

    global _ZONE_PAIR_MAP, _ZONE_TIME_MAP, _GLOBAL_MEAN
    _GLOBAL_MEAN = float(train["duration_seconds"].mean())

    _ZONE_PAIR_MAP = (
        train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"]
        .mean().to_dict()
    )
    _ZONE_TIME_MAP = (
        train.groupby(["pickup_zone", "dropoff_zone", "_hour"])["duration_seconds"]
        .mean().to_dict()
    )

    # Save for use in predict.py
    import pickle
    zone_maps = {
        "zone_pair": _ZONE_PAIR_MAP,
        "zone_time": _ZONE_TIME_MAP,
        "global_mean": _GLOBAL_MEAN,
    }
    with open(Path(__file__).parent / "zone_maps.pkl", "wb") as f:
        pickle.dump(zone_maps, f)
    print(f"Saved zone_maps.pkl ({len(_ZONE_PAIR_MAP)} pairs, {len(_ZONE_TIME_MAP)} time entries)")
    
    X_train = engineer_features(train)
    y_train = train["duration_seconds"].to_numpy()
    X_dev = engineer_features(dev)
    y_dev = dev["duration_seconds"].to_numpy()

    if SKIP_TRAIN and MODEL_PATH.exists():
        print("\nSkipping training — loading existing model...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        print("\nTraining XGBoost...")
        model = xgb.XGBRegressor(
            n_estimators=800,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        t0 = time.time()
        model.fit(X_train, y_train, verbose=False)
        print(f"  trained in {time.time() - t0:.0f}s")

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved model to {MODEL_PATH}")

    if not SKIP_TRAIN:
        preds = model.predict(X_dev)
        mae = float(np.mean(np.abs(preds - y_dev)))
        print(f"\nDev MAE: {mae:.1f} seconds")


if __name__ == "__main__":
    main()
