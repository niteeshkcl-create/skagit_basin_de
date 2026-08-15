import xarray as xr
import pandas as pd
import numpy as np
import json
from pathlib import Path
from shapely.geometry import shape
import geopandas as gpd
import regionmask

URL = "s3://daymet-zarr/north_america/na_daily.zarr"
BOUNDARY = Path("data/GIS/SkagitBoundary.json")
OUTPUT_CSV = Path("data/weather_data/daymet_skagit_precip_daily_basin_full.csv")

def extract():
    print(f"Opening Daymet Zarr from {URL}...")
    try:
        ds = xr.open_zarr(URL, consolidated=True, storage_options={'anon': True})
    except Exception as e:
        print(f"Failed to open Zarr: {e}")
        # Try alternate path
        URL_ALT = "s3://daymet-zarr/daily/na.zarr"
        print(f"Trying {URL_ALT}...")
        ds = xr.open_zarr(URL_ALT, consolidated=True, storage_options={'anon': True})

    print("Building mask...")
    with open(BOUNDARY) as f:
        geom = json.load(f)
        if geom.get("type") == "FeatureCollection":
            geom = geom["features"][0]["geometry"]
    
    poly = shape(geom)
    gdf = gpd.GeoDataFrame({"name": ["Skagit"]}, geometry=[poly], crs="EPSG:4326")
    
    mask = regionmask.mask_geopandas(gdf, ds.lon, ds.lat)
    mask = ~mask.isnull()
    
    # Slice spatially
    y_idx = np.where(mask.any(dim="x"))[0]
    x_idx = np.where(mask.any(dim="y"))[0]
    y_slice = slice(y_idx.min(), y_idx.max() + 1)
    x_slice = slice(x_idx.min(), x_idx.max() + 1)
    
    ds_sub = ds.isel(y=y_slice, x=x_slice)
    mask_sub = mask.isel(y=y_slice, x=x_slice)
    
    weights = mask_sub.astype("float32")
    weights = weights / weights.sum()
    
    print("Computing basin average series...")
    # Variable is 'prcp' in Daymet
    p = ds_sub['prcp']
    basin_series = (p * weights).sum(['y', 'x'], skipna=True).compute().to_series()
    
    print(f"Saving to {OUTPUT_CSV}...")
    basin_series.to_csv(OUTPUT_CSV)
    print("Done!")

if __name__ == "__main__":
    extract()
