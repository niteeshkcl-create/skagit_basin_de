import xarray as xr
import pandas as pd
import numpy as np
import json
from pathlib import Path
from shapely.geometry import shape
import geopandas as gpd
import regionmask

BOUNDARY = Path("data/GIS/SkagitBoundary.json")
OUTPUT_CSV = Path("data/weather_data/daymet_skagit_precip_daily_basin_backfill.csv")

def extract_year(year):
    url = f"https://thredds.daac.ornl.gov/thredds/dodsC/ornldaac/2129/daymet_v4_daily_na_prcp_{year}.nc"
    print(f"Opening Daymet {year} from {url}...")
    ds = xr.open_dataset(url)
    
    with open(BOUNDARY) as f:
        geom = json.load(f)
        if geom.get("type") == "FeatureCollection":
            geom = geom["features"][0]["geometry"]
    
    poly = shape(geom)
    gdf = gpd.GeoDataFrame({"name": ["Skagit"]}, geometry=[poly], crs="EPSG:4326")
    
    # Get mask (only once)
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
    
    p = ds_sub['prcp']
    basin_series = (p * weights).sum(['y', 'x'], skipna=True).compute().to_series()
    return basin_series

def main():
    years = [2012, 2013, 2020, 2021, 2022, 2023, 2024]
    all_series = []
    for y in years:
        try:
            ser = extract_year(y)
            all_series.append(ser)
        except Exception as e:
            print(f"Failed for {y}: {e}")
    
    if all_series:
        final = pd.concat(all_series).sort_index()
        final.to_csv(OUTPUT_CSV)
        print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
