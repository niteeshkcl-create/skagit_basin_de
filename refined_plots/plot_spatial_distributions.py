import os
import sys
import geopandas as gpd
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as dt, timedelta
from zipfile import ZipFile
import rioxarray  # noqa: F401

BASE_DIR = "/home/nksp2/skagit/skagit_2/skagit-met"
BOUNDARY_PATH = os.path.join(BASE_DIR, "data/GIS/SkagitBoundary.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "refined_plots/spatial")
TEMP_DATA_DIR = os.path.join(BASE_DIR, "refined_plots/temp_prism")

BASE_URL = "https://services.nacse.org/prism/data/get"
REGION = "us"
RESOLUTION = "4km"
PARAMS = ["ppt", "tmean"]

# Peaks list for the big events
TARGET_EVENTS = [
    {"date": "1990-11-11", "scale": 4},
    {"date": "1990-11-25", "scale": 5},
    {"date": "1995-11-30", "scale": 4},
    {"date": "2003-10-22", "scale": 3},
    {"date": "2006-11-07", "scale": 5},
    {"date": "2021-11-15", "scale": 4},
]

def download_prism_file(var, date_str, output_dir):
    """Download a single PRISM daily file. Returns path or None."""
    url = f"{BASE_URL}/{REGION}/{RESOLUTION}/{var}/{date_str}"
    try:
        resp = requests.get(url, params={"format": "nc"}, stream=True, timeout=120)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").lower()
        fname = f"{var}_{date_str}_{RESOLUTION}"
        out_path = os.path.join(output_dir, fname + (".zip" if "zip" in content_type else ".nc"))
        
        # Avoid redownloading if file already exists
        if os.path.exists(out_path):
            return out_path
            
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"  Downloaded: {var} {date_str}")
        return out_path
    except Exception as e:
        print(f"  [WARN] Failed to download {var} {date_str}: {e}")
        return None

def build_prism_dataset(file_paths, boundary_gdf):
    """Load, clip to Skagit boundary, and merge PRISM files into an xr.Dataset."""
    rasters = []
    for path in filter(None, file_paths):
        # Handle zip
        if path.endswith(".zip"):
            with ZipFile(path, "r") as archive:
                nc_members = [m for m in archive.namelist() if m.endswith(".nc")]
                if not nc_members:
                    continue
                archive.extract(nc_members[0], path=os.path.dirname(path))
                path = os.path.join(os.path.dirname(path), nc_members[0])

        fname = os.path.basename(path)
        parts = fname.replace(".nc", "").replace(".zip", "").split("_")
        variable = None
        date_part = None
        for part in parts:
            if len(part) == 8 and part.isdigit():
                date_part = part
        known_vars = {"ppt", "tmean", "tmax", "tmin", "vpdmax", "vpdmin", "tdmean"}
        for p in parts:
            if p in known_vars:
                variable = p
                break
        if variable is None:
            variable = parts[0]
        if date_part is None:
            date_part = "".join(c for c in fname if c.isdigit())[:8]

        try:
            ds = xr.open_dataset(path, engine="netcdf4")
            ds = ds.drop_vars(["crs", "spatial_ref"], errors="ignore")
            var_name = variable if variable in ds.data_vars else list(ds.data_vars)[0]
            da = ds[var_name]

            # Rename spatial dims
            rename = {}
            for old, new in [("x", "lon"), ("y", "lat"), ("longitude", "lon"), ("latitude", "lat")]:
                if old in da.dims:
                    rename[old] = new
            if rename:
                da = da.rename(rename)

            date = dt.strptime(date_part, "%Y%m%d")
            da = da.expand_dims(dim="time").assign_coords(time=("time", [date]))

            # Clip to basin
            da = da.rio.write_crs("EPSG:4326", inplace=False)
            da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
            da = da.rio.clip(boundary_gdf.to_crs("EPSG:4326").geometry, drop=True)

            rasters.append(da.rename(variable))
        except Exception as e:
            print(f"  [WARN] Could not process {path}: {e}")
            continue

    if not rasters:
        return None
    return xr.merge(rasters)

def plot_spatial_event(peak_date_str, ar_scale, boundary_gdf):
    peak_date = dt.strptime(peak_date_str, "%Y-%m-%d")
    event_id = peak_date.strftime("%Y-%m-%d")
    start_date = peak_date - timedelta(days=3)
    end_date = peak_date + timedelta(days=3)
    dates = pd.date_range(start_date, end_date, freq="1D").strftime("%Y%m%d").tolist()

    print(f"\n=== Event {event_id} (AR Scale {ar_scale}) ===")
    print(f"    Window: {start_date.date()} → {end_date.date()}")

    event_temp_dir = os.path.join(TEMP_DATA_DIR, event_id)
    os.makedirs(event_temp_dir, exist_ok=True)

    # Parallel download
    tasks = [(var, date) for var in PARAMS for date in dates]
    with ThreadPoolExecutor(max_workers=6) as ex:
        file_paths = list(ex.map(lambda args: download_prism_file(*args, event_temp_dir), tasks))

    ds = build_prism_dataset(file_paths, boundary_gdf)
    if ds is None:
        print(f"  [SKIP] No data built for {event_id}")
        return

    # Metrics
    storm_ppt = ds["ppt"].sum(dim="time") if "ppt" in ds else None
    storm_tmean = ds["tmean"].mean(dim="time") if "tmean" in ds else None

    fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    fig.patch.set_facecolor("#f4f4f4")

    # Precipitation panel
    ax1 = axes[0]
    if storm_ppt is not None and not np.all(np.isnan(storm_ppt.values)):
        vmax = float(np.nanpercentile(storm_ppt.values, 99))
        im1 = ax1.pcolormesh(storm_ppt.lon, storm_ppt.lat, storm_ppt.values,
                             transform=ccrs.PlateCarree(),
                             cmap="YlGnBu", vmin=0, vmax=max(vmax, 1))
        plt.colorbar(im1, ax=ax1, label="Total Precipitation (mm)", shrink=0.7)
    else:
        ax1.text(0.5, 0.5, "No PPT Data", transform=ax1.transAxes, ha="center")

    ax1.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax1.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.6)
    ax1.add_feature(cfeature.STATES, linestyle="--", linewidth=0.5)
    boundary_gdf.plot(ax=ax1, facecolor="none", edgecolor="red", lw=2.5, transform=ccrs.PlateCarree())
    ax1.set_title(f"Cumulative Precipitation\n{start_date.strftime('%b %d')} – {end_date.strftime('%b %d, %Y')}",
                  fontsize=12, fontweight="bold")

    # Temperature panel
    ax2 = axes[1]
    if storm_tmean is not None and not np.all(np.isnan(storm_tmean.values)):
        vabs = float(np.nanmax(np.abs(storm_tmean.values)))
        im2 = ax2.pcolormesh(storm_tmean.lon, storm_tmean.lat, storm_tmean.values,
                             transform=ccrs.PlateCarree(),
                             cmap="RdBu_r", vmin=-vabs, vmax=vabs)
        plt.colorbar(im2, ax=ax2, label="Mean Temperature (°C)", shrink=0.7)
    else:
        ax2.text(0.5, 0.5, "No TMEAN Data", transform=ax2.transAxes, ha="center")

    ax2.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax2.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.6)
    ax2.add_feature(cfeature.STATES, linestyle="--", linewidth=0.5)
    boundary_gdf.plot(ax=ax2, facecolor="none", edgecolor="black", lw=2.5, transform=ccrs.PlateCarree())
    ax2.set_title(f"Mean Temperature\n{start_date.strftime('%b %d')} – {end_date.strftime('%b %d, %Y')}",
                  fontsize=12, fontweight="bold")

    plt.suptitle(f"Skagit Basin – Major AR Event: {event_id}  |  AR Scale: {ar_scale}",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_file = os.path.join(OUTPUT_DIR, f"event_{event_id}_spatial.png")
    plt.savefig(out_file, bbox_inches="tight", dpi=150)
    print(f"  ✓ Saved: {out_file}")
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DATA_DIR, exist_ok=True)

    boundary_gdf = gpd.read_file(BOUNDARY_PATH)
    print(f"Generating spatial distribution maps for {len(TARGET_EVENTS)} big AR events...")

    for ev in TARGET_EVENTS:
        try:
            plot_spatial_event(ev["date"], ev["scale"], boundary_gdf)
        except Exception as e:
            print(f"  [ERROR] {ev['date']}: {e}")

    print("\nAll spatial maps complete.")

if __name__ == "__main__":
    main()
