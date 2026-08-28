import os
import pandas as pd
import xarray as xr
import numpy as np
import geopandas as gpd
import regionmask
import zipfile
import tempfile
import rioxarray

BASE_DIR = "/data0/hernanqd/plots_code/skagit_basin_de"
HUC8_GEO = os.path.join(BASE_DIR, "data/GIS/SkagitSubBasin_HUC8.geojson")

# Paths to compare
DATA_TRANSFER_4K_ZIP = "/data0/skagit_met/data_transfer/data/PRISM/2001-2010/ppt_20030101_4km.zip"
NEWER_25M_ZIP = "/data0/hernanqd/plots_code/skagit_basin_de/cumulative_precipitation_plot/exploring/prism_ppt_us_25m_20030101.zip"

def load_regions():
    """Load subbasin geometries"""
    gdf = gpd.read_file(HUC8_GEO).to_crs("EPSG:4326")
    return gdf

def load_from_zip(zip_path):
    """Extract and load data from zip file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmpdir)
            # Look for data files
            for root, dirs, files in os.walk(tmpdir):
                for f in files:
                    if f.endswith('.nc'):
                        filepath = os.path.join(root, f)
                        ds = xr.open_dataset(filepath)
                        return ds.copy(deep=True), 'netcdf'
                    elif f.endswith(('.tif', '.tiff')):
                        filepath = os.path.join(root, f)
                        da = rioxarray.open_rasterio(filepath)
                        return da, 'geotiff'
    return None, None

def get_mask(gdf, lon, lat):
    """Create basin masks following 3_bulk_bias.py approach"""
    mask = regionmask.mask_3D_geopandas(gdf, lon, lat)
    region_indices = []
    region_names = ["Upper Skagit", "Sauk", "Lower Skagit"]
    for r in region_names:
        idx = gdf[gdf["Name"].str.contains(r, case=False)].index[0]
        region_indices.append(idx)
    return mask.sel(region=region_indices), region_names

def calculate_basin_mean(da, mask_2d):
    """Calculate mean for a basin following 3_bulk_bias.py"""
    data = da.values
    mask = mask_2d.values if hasattr(mask_2d, 'values') else mask_2d

    # Handle different dimensionalities
    if data.ndim == 3:  # (band, lat, lon) or (time, lat, lon)
        data = data[0]  # Take first band/time

    if data.ndim == 2:
        # Mask out nodata values (typically -9999 or negative values)
        data = data.astype(float)
        data[data < 0] = np.nan  # Treat negative values as nodata

        data_flat = data.flatten()
        mask_flat = mask.flatten()
        mask_idx = mask_flat > 0
        if mask_idx.any():
            return np.nanmean(data_flat[mask_idx])

    return np.nan

def compare_versions():
    """Main comparison function"""
    print("Loading subbasin geometry...")
    gdf = load_regions()

    # Load 4km version
    print(f"\nLoading 4km version from zip: {DATA_TRANSFER_4K_ZIP}")
    data_4k, fmt_4k = load_from_zip(DATA_TRANSFER_4K_ZIP)

    if data_4k is None:
        print("Error: Could not load 4km zip file")
        return

    # Extract data and coordinates
    if fmt_4k == 'geotiff':
        print(f"✓ Loaded as GeoTIFF")
        da_4k = data_4k.squeeze()
        lon_4k = data_4k.x.values
        lat_4k = data_4k.y.values
        ppt_label = "Band1"
    else:  # NetCDF
        print(f"✓ Loaded as NetCDF")
        # Find the precipitation variable (ppt, Band1, etc.)
        ppt_vars = [v for v in data_4k.data_vars if v not in ['crs']]
        ppt_var = ppt_vars[0] if ppt_vars else 'ppt'
        da_4k = data_4k[ppt_var]
        if da_4k.ndim == 0:  # scalar
            # Try Band1
            if 'Band1' in data_4k.data_vars:
                da_4k = data_4k['Band1']
                ppt_label = "Band1"
        ppt_label = ppt_var
        lon_4k = data_4k.lon.values
        lat_4k = data_4k.lat.values

    # Handle 3D arrays (band, lat, lon)
    if da_4k.ndim == 3:
        da_4k = da_4k.squeeze('band') if 'band' in da_4k.dims else da_4k.isel(dim_0=0)

    print(f"  Variable: {ppt_label}")
    print(f"  Shape: {da_4k.shape}")
    print(f"  Data range: [{float(da_4k.min()):.2f}, {float(da_4k.max()):.2f}] mm")

    # Get mask for 4km grid
    mask_4k, subbasin_names = get_mask(gdf, lon_4k, lat_4k)

    # Calculate means for each subbasin
    print("\n4km PRISM Basin Means:")
    results_4k = {}
    for i, name in enumerate(subbasin_names):
        mask_2d = mask_4k.isel(region=i)
        mean_val = calculate_basin_mean(da_4k, mask_2d)
        results_4k[name] = mean_val
        print(f"  {name:20s}: {mean_val:.4f} mm")

    # Load 25m version
    print(f"\nLoading 25m version from zip: {NEWER_25M_ZIP}")
    data_25m, fmt_25m = load_from_zip(NEWER_25M_ZIP)

    if data_25m is None:
        print("Error: Could not load 25m zip file")
        return

    # Extract data and coordinates
    if fmt_25m == 'geotiff':
        print(f"✓ Loaded as GeoTIFF")
        da_25m = data_25m.squeeze()
        lon_25m = data_25m.x.values
        lat_25m = data_25m.y.values
        ppt_label_25m = "Band1"
    else:  # NetCDF
        print(f"✓ Loaded as NetCDF")
        ppt_vars = [v for v in data_25m.data_vars if v not in ['crs']]
        ppt_var = ppt_vars[0] if ppt_vars else 'ppt'
        da_25m = data_25m[ppt_var]
        if da_25m.ndim == 0:  # scalar
            if 'Band1' in data_25m.data_vars:
                da_25m = data_25m['Band1']
                ppt_label_25m = "Band1"
        ppt_label_25m = ppt_var
        lon_25m = data_25m.lon.values
        lat_25m = data_25m.lat.values

    # Handle 3D arrays
    if da_25m.ndim == 3:
        da_25m = da_25m.squeeze('band') if 'band' in da_25m.dims else da_25m.isel(dim_0=0)

    print(f"  Variable: {ppt_label_25m}")
    print(f"  Shape: {da_25m.shape}")
    print(f"  Data range: [{float(da_25m.min()):.2f}, {float(da_25m.max()):.2f}] mm")

    # Get mask for 25m grid
    mask_25m, _ = get_mask(gdf, lon_25m, lat_25m)

    # Calculate means for each subbasin
    print("\n25m PRISM Basin Means:")
    results_25m = {}
    for i, name in enumerate(subbasin_names):
        mask_2d = mask_25m.isel(region=i)
        mean_val = calculate_basin_mean(da_25m, mask_2d)
        results_25m[name] = mean_val
        print(f"  {name:20s}: {mean_val:.4f} mm")

    # Compare
    print("\n" + "="*80)
    print("COMPARISON: 4km vs 25m PRISM (2003-01-01)")
    print("="*80)
    print(f"{'Subbasin':<20s} {'4km (mm)':<15s} {'25m (mm)':<15s} {'Difference':<15s} {'% Diff':<10s}")
    print("-" * 80)
    for name in subbasin_names:
        val_4k = results_4k.get(name, np.nan)
        val_25m = results_25m.get(name, np.nan)
        if not np.isnan(val_4k) and not np.isnan(val_25m):
            diff = val_25m - val_4k
            pct_diff = (diff / val_4k * 100) if val_4k != 0 else np.nan
            print(f"{name:<20s} {val_4k:<15.4f} {val_25m:<15.4f} {diff:<15.4f} {pct_diff:<10.2f}%")
        else:
            print(f"{name:<20s} {'ERROR':<15s} {'ERROR':<15s} {'N/A':<15s} {'N/A':<10s}")

if __name__ == "__main__":
    compare_versions()
