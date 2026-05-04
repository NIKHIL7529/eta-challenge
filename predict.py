"""Submission interface — this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. The signature below
is fixed; everything else (model type, preprocessing, etc.) is yours to change.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
import math

import numpy as np

_MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _MODEL = pickle.load(_f)

_ZONE_MAPS_PATH = Path(__file__).parent / "zone_maps.pkl"
with open(_ZONE_MAPS_PATH, "rb") as f:
    _ZONE_MAPS = pickle.load(f)

_ZONE_PAIR_MAP = _ZONE_MAPS["zone_pair"]
_ZONE_TIME_MAP = _ZONE_MAPS["zone_time"]
_GLOBAL_MEAN   = _ZONE_MAPS["global_mean"]

_ZONE_COORDS_PATH = Path(__file__).parent / "zone_coords.pkl"

with open(_ZONE_COORDS_PATH, "rb") as f:
    _ZONE_COORDS = pickle.load(f)
# Disable xgboost's feature-name validation so we can predict on a bare
# numpy array (skips per-call DataFrame construction overhead).
if hasattr(_MODEL, "get_booster"):
    _MODEL.get_booster().feature_names = None

# Feature order must match baseline.py:
#   pickup_zone, dropoff_zone, hour, dow, month, passenger_count, distance,
#   zone_pair_mean, zone_time_mean

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def predict(request: dict) -> float:
    """Predict trip duration in seconds.

    Input schema:
        {
            "pickup_zone":     int,   # NYC taxi zone, 1-265
            "dropoff_zone":    int,
            "requested_at":    str,   # ISO 8601 datetime
            "passenger_count": int,
        }
    """
    ts = datetime.fromisoformat(request["requested_at"])
    pickup = int(request["pickup_zone"])
    drop = int(request["dropoff_zone"])
    hour = ts.hour

    lat1, lon1 = _ZONE_COORDS.get(pickup, (0.0, 0.0))
    lat2, lon2 = _ZONE_COORDS.get(drop, (0.0, 0.0))
    distance = haversine(lat1, lon1, lat2, lon2)

    zone_pair_mean = _ZONE_PAIR_MAP.get((pickup, drop), _GLOBAL_MEAN)
    zone_time_mean = _ZONE_TIME_MAP.get((pickup, drop, hour), zone_pair_mean)

    x = np.array([[
        pickup,
        drop,
        hour,
        ts.weekday(),
        ts.month,
        int(request["passenger_count"]),
        distance,
        zone_pair_mean,
        zone_time_mean,
    ]], dtype=np.float32)

    model_pred = float(_MODEL.predict(x)[0])

    # Blending: zone signal dominates, model adds generalization
    if (pickup, drop, hour) in _ZONE_TIME_MAP:
        return 0.7 * zone_time_mean + 0.3 * model_pred
    if (pickup, drop) in _ZONE_PAIR_MAP:
        return 0.7 * zone_pair_mean + 0.3 * model_pred
    return model_pred
