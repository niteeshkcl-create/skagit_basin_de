import intake
import pandas as pd
import xarray as xr
import numpy as np
import json
from pathlib import Path
import geopandas as gpd
import regionmask
from shapely.geometry import shape

CAT_URL = "https://raw.githubusercontent.com/hytest-org/hytest/main/dataset_catalog/hytest_intake_catalog.yml"
BOUNDARY = Path("data/GIS/SkagitBoundary.json")
OUTPUT_CSV = Path("data/weather_data/conus404_skagit_precip_daily_basin_only.csv")

def extract():
    print("Opening catalog...")
    cat = intake.open_catalog(CAT_URL)
    ds = cat['conus404-catalog']['conus404-daily-osn'].to_dask()
    
    print("Building mask...")
    with open(BOUNDARY) as f:
        geom = json.load(f)
        if geom.get("type") == "FeatureCollection":
            geom = geom["features"][0]["geometry"]
        elif geom.get("type") == "Feature":
            geom = geom["geometry"]
    
    poly = shape(geom)
    gdf = gpd.GeoDataFrame({"name": ["Skagit"]}, geometry=[poly], crs="EPSG:4326")
    
    # Get lat/lon
    lon = ds['lon']
    lat = ds['lat']
    if "time" in lon.dims: lon = lon.isel(time=0, drop=True)
    if "time" in lat.dims: lat = lat.isel(time=0, drop=True)
    
    mask = regionmask.mask_geopandas(gdf, lon, lat)
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
    # Select full time range (1979-2024)
    p = ds_sub['PREC_ACC_NC']
    basin_series = (p * weights).sum(['y', 'x'], skipna=True).compute().to_series()
    
    print(f"Saving to {OUTPUT_CSV}...")
    basin_series.to_csv(OUTPUT_CSV)
    print("Done!")

if __name__ == "__main__":
    extract()
