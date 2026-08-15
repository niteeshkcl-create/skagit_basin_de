import argparse
import os
import pathlib
from datetime import datetime as dt
import geopandas as gpd
import pandas as pd
import requests
import rioxarray
import xarray as xr
from concurrent.futures import ThreadPoolExecutor

# gridMET URL pattern for precipitation (daily)
BASE_URL = "http://www.northwestknowledge.net/metdata/data"
DEFAULT_GEOJSON = "data/GIS/SkagitBoundary.json"
DEFAULT_OUTPUT_DIR = "/data0/skagit_met/data_transfer/data/gridmet/"

def setup_args():
    parser = argparse.ArgumentParser(description="Download gridMET daily precipitation and clip to Skagit basin.")
    parser.add_argument("--years", type=int, nargs="+", required=True, help="Years to download (e.g. 2003 2006 2016 2018 2021)")
    parser.add_argument("--outputDir", default=DEFAULT_OUTPUT_DIR, help="Directory to save Zarr output.")
    parser.add_argument("--geojson", default=DEFAULT_GEOJSON, help="GeoJSON file for clipping.")
    return parser.parse_args()

def download_gridmet_year(year, output_dir):
    filename = f"pr_{year}.nc"
    url = f"{BASE_URL}/{filename}"
    local_path = os.path.join(output_dir, filename)
    
    if os.path.exists(local_path):
        print(f"[SKIP] {filename} already exists.")
        return local_path
        
    print(f"[DOWNLOAD] {url} -> {local_path}")
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_path
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        return None

def process_to_zarr(nc_path, year, boundaries_gdf, output_dir):
    if not nc_path:
        return
    
    zarr_name = f"{year}_daily_4km_gridMET_data.zarr"
    zarr_path = os.path.join(output_dir, zarr_name)
    
    print(f"[PROCESS] {nc_path} -> {zarr_path}")
    try:
        # gridMET uses 'precipitation_amount' as the variable name for 'pr' files
        ds = xr.open_dataset(nc_path, engine="netcdf4")
        
        # Rename coords to match existing project conventions if necessary
        # gridMET usually has 'lon', 'lat', 'day' (time)
        if 'day' in ds.coords:
            ds = ds.rename({'day': 'time'})
            
        # Select precipitation variable
        # Gridmet pr variable name is 'precipitation_amount'
        var_name = 'precipitation_amount'
        if var_name not in ds.data_vars:
            # Fallback to identify the precip variable
            var_name = [v for v in ds.data_vars if 'precip' in v.lower()][0]

        da = ds[var_name]
        
        # Ensure CRS and clip
        da = da.rio.write_crs("EPSG:4326")
        clipped = da.rio.clip(boundaries_gdf.geometry, boundaries_gdf.crs)
        
        # Rename variable to 'ppt' or 'prcp' to align with Daymet/PRISM in the project
        # In AR_Events.ipynb, PRISM is 'ppt', Daymet is 'prcp'. 
        # I'll use 'prcp' as it's common.
        clipped = clipped.to_dataset(name='prcp')
        
        # Save to Zarr
        clipped.to_zarr(zarr_path, mode="w")
        print(f"[SUCCESS] Saved {zarr_path}")
        
        # Optional: remove NC file to save space
        # os.remove(nc_path)
        
    except Exception as e:
        print(f"[ERROR] Failed to process {nc_path}: {e}")

if __name__ == "__main__":
    args = setup_args()
    os.makedirs(args.outputDir, exist_ok=True)
    
    boundaries = gpd.read_file(args.geojson).to_crs("EPSG:4326")
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        nc_paths = list(executor.map(lambda y: download_gridmet_year(y, args.outputDir), args.years))
    
    for i, year in enumerate(args.years):
        process_to_zarr(nc_paths[i], year, boundaries, args.outputDir)
