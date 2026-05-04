## Final Architecture (as submitted)

### Files
- model.pkl: XGBoost regressor, 9 features
- zone_maps.pkl: {"zone_pair": dict, "zone_time": dict, "global_mean": float}
- zone_coords.pkl: {zone_id: (lat, lon)} for 265 NYC taxi zones

### Feature list (order matters, must match baseline.py FEATURES)
1. pickup_zone
2. dropoff_zone  
3. hour
4. dow (day of week)
5. month
6. passenger_count
7. distance (haversine km between zone centroids)
8. zone_pair_mean (historical avg for this pickup/dropoff pair)
9. zone_time_mean (historical avg for this pickup/dropoff/hour triple)

### Inference blending logic
if (pickup, dropoff, hour) in zone_time_map:
    return 0.7 * zone_time_mean + 0.3 * model_pred
if (pickup, dropoff) in zone_pair_map:
    return 0.7 * zone_pair_mean + 0.3 * model_pred
return model_pred

### Dev MAE progression
356s → 304s → 295s → 276s → 275s → 271s