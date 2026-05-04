# ETA Challenge - Gobblecube Submission

## Final score
Dev MAE: **~271 s** (baseline: 356 s, improvement: ~24%)

---

## Approach

Built a hybrid predictor combining precomputed zone-pair lookup tables with 
an XGBoost regressor. The core insight - that (pickup_zone, dropoff_zone) 
historical averages alone outperform the GBT baseline - came immediately from 
reading the README carefully. I extended this to time-aware lookups keyed on 
(pickup_zone, dropoff_zone, hour), which captured rush-hour and off-peak 
traffic patterns. Zone centroid distances (haversine from the TLC shapefile) 
and the lookup values themselves were added as model features. At inference, 
a hierarchical blending strategy combines the lookup with model output: 
time-aware lookup dominates (0.7 weight) when the route+hour key exists, 
zone-pair lookup dominates (0.7) as fallback, and the model handles 
genuinely unseen routes.

---

## What didn't work

**Count-based filtering on time-aware averages:** Tried requiring ≥10, ≥15, 
≥20 samples before trusting a zone+hour average. Negligible improvement 
(276.7 → 277.5 MAE depending on threshold) while adding complexity. Dropped it 
- the data is dense enough that sparse pairs aren't a real problem.

**Physics-based speed prior:** Built a feature using distance / average_speed_per_hour. 
Degraded MAE from ~275 to ~275.9. Zone-time averages already encode traffic 
implicitly - adding a simplified speed model introduced noise rather than signal.

**Model-only without blending:** After injecting zone_pair_mean and zone_time_mean 
as model features, tried removing the lookup blending entirely and letting XGBoost 
learn everything. MAE worsened to ~293. Tree models at reasonable depth can't 
fully memorize 265×265×24 interaction patterns - blending is necessary.

---

## Where AI tooling helped

Used ChatGPT throughout for iteration speed, not for reasoning. Specific wins:

- Generated the haversine + shapefile centroid extraction code
- Wrote the hierarchical fallback blending logic after I described the structure
- Debugged the NaN MAE issue (time-aware lookup returning NaN when fallback 
  wasn't wired correctly)

Where it fell short: blend weight tuning required manual experimentation across 
~15 combinations. ChatGPT suggested weights but couldn't predict which would 
work without running grade.py. The decision to keep blending vs go model-only 
required understanding why tree models fail on high-cardinality lookups - 
ChatGPT explained this correctly when asked, but didn't flag it proactively.

---

## Next experiments

**Holiday features (highest priority):** The README explicitly says the Eval 
set is a "winter-holiday slice." Adding is_holiday and is_eve flags from a 
stored date set would directly target the Dev→Eval gap without touching the 
model architecture.

**NOAA weather join:** The README lists this as a suggested dataset. 
Precipitation and temperature as hourly features joined to requested_at. 
Didn't implement due to pipeline complexity and time.

**Learned blending weights:** Currently fixed weights (0.7/0.3). A meta-model 
trained on residuals could learn when to trust the lookup vs the model - 
likely meaningful for sparse routes.

**Zone embeddings:** Representing zones as learned vectors rather than integer 
IDs. Would help generalization for low-frequency zone pairs.

---

## How to reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# If zone_coords.pkl is missing, regenerate it:
python data/compute_zone_centroids.py

# 1. Data (one-time, ~500 MB)
python data/download_data.py

# 2. Train (produces model.pkl + zone_maps.pkl)
python baseline.py

# 3. Score locally
python grade.py

# 4. Run contract tests
python -m pytest tests/ -v

# 5. Docker verification
docker build -t my-eta .
docker run --rm -v $(pwd)/data:/work my-eta /work/dev.parquet /work/preds.csv
```

---

Total time spent: ~12 hours