import shapefile
from shapely.geometry import shape
from pathlib import Path
import pickle

shp_path = Path("data/taxi_zones.shp")

sf = shapefile.Reader(str(shp_path))

zone_coords = {}

for sr in sf.shapeRecords():
    geom = shape(sr.shape.__geo_interface__)
    centroid = geom.centroid

    zone_id = int(sr.record["LocationID"])
    zone_coords[zone_id] = (centroid.y, centroid.x)

with open("zone_coords.pkl", "wb") as f:
    pickle.dump(zone_coords, f)

print(f"Saved {len(zone_coords)} zone coordinates")