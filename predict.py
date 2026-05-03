"""Submission interface — this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. The signature below
is fixed; everything else (model type, preprocessing, etc.) is yours to change.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

_MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _MODEL = pickle.load(_f)

_ZONE_PATH = Path(__file__).parent / "zone_avg.pkl"

with open(_ZONE_PATH, "rb") as f:
    _ZONE_AVG = pickle.load(f)
# Disable xgboost's feature-name validation so we can predict on a bare
# numpy array (skips per-call DataFrame construction overhead).
if hasattr(_MODEL, "get_booster"):
    _MODEL.get_booster().feature_names = None

# Feature order must match baseline.py:
#   pickup_zone, dropoff_zone, hour, dow, month, passenger_count


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
    x = np.array(
        [[
            int(request["pickup_zone"]),
            int(request["dropoff_zone"]),
            ts.hour,
            ts.weekday(),
            ts.month,
            int(request["passenger_count"]),
        ]],
        dtype=np.int32,
    )
    key = (int(request["pickup_zone"]), int(request["dropoff_zone"]))

    # Use zone average if available
    if key in _ZONE_AVG:
        zone_pred = float(_ZONE_AVG[key])
        model_pred = float(_MODEL.predict(x)[0])
        return 0.7 * zone_pred + 0.3 * model_pred

    # fallback to model
    return float(_MODEL.predict(x)[0])
