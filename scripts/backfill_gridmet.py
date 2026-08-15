import xarray as xr
import pandas as pd
import numpy as np
import json
from pathlib import Path
from shapely.geometry import shape
import geopandas as gpd
import regionmask

BOUNDARY = Path("data/GIS/SkagitBoundary.json")
OUTPUT_CSV = Path("data/weather_data/gridmet_skagit_precip_daily_basin_backfill.csv")
URL = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_pr_1979_CurrentYear_CONUS.nc"

def main():
    print(f"Opening GridMET aggregation from {URL}...")
    ds = xr.open_dataset(URL, chunks={'day': 500})
    
    with open(BOUNDARY) as f:
        geom = json.load(f)
        if geom.get("type") == "FeatureCollection":
            geom = geom["features"][0]["geometry"]
    
    poly = shape(geom)
    gdf = gpd.GeoDataFrame({"name": ["Skagit"]}, geometry=[poly], crs="EPSG:4326")
    
    # GridMET uses 'lon' and 'lat'
    mask = regionmask.mask_geopandas(gdf, ds.lon, ds.lat)
    mask = ~mask.isnull()
    
    # Slice spatially
    y_idx = np.where(mask.any(dim="lon"))[0]
    x_idx = np.where(mask.any(dim="lat"))[0]
    y_slice = slice(y_idx.min(), y_idx.max() + 1)
    x_slice = slice(x_idx.min(), x_idx.max() + 1)
    
    ds_sub = ds.isel(lat=y_slice, lon=x_slice)
    mask_sub = mask.isel(lat=y_slice, lon=x_slice)
    
    weights = mask_sub.astype("float32")
    weights = weights / weights.sum()
    
    print("Computing basin average series (2014-Present)...")
    # GridMET has 'precipitation_amount'
    p = ds_sub['precipitation_amount'].sel(day=slice("1981-01-01", "2025-12-31"))
    basin_series = (p * weights).sum(['lat', 'lon'], skipna=True).compute().to_series()
    
    print(f"Saving to {OUTPUT_CSV}...")
    basin_series.to_csv(OUTPUT_CSV)
    print("Done!")

if __name__ == "__main__":
    main()
