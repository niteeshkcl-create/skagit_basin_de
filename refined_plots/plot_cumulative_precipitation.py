import sys
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Add scratch path for ar_events_lib
sys.path.append("/home/nksp2/skagit/skagit_2/skagit-met/scratch")
import ar_events_lib as lib

# --- CONFIGURATION & PATHS ---
BASE_DIR = "/home/nksp2/skagit/skagit_2/skagit-met"
SNOTEL_DIR = os.path.join(BASE_DIR, "data/weather_data")
OUT_DIR = os.path.join(BASE_DIR, "refined_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# 5 Major Atmospheric River events with 2-week windows (excluding AR3 and AR2, adding 1990 and 1995)
EVENTS = [
    {"year": 1990, "start": "1990-11-03", "end": "1990-11-17", "label": "Nov 3–17, 1990 (AR4)"},
    {"year": 1995, "start": "1995-11-22", "end": "1995-12-06", "label": "Nov 22–Dec 6, 1995 (AR4)"},
    {"year": 2003, "start": "2003-10-18", "end": "2003-10-31", "label": "Oct 18–31, 2003 (AR5)"},
    {"year": 2006, "start": "2006-11-04", "end": "2006-11-17", "label": "Nov 4–17, 2006 (AR4)"},
    {"year": 2021, "start": "2021-11-10", "end": "2021-11-23", "label": "Nov 10–23, 2021 (AR5)"},
]

# Grid Colors matching standard palette (removed ORNL)
GRID_COLORS = {
    "PRISM 4km":     "#1B9E77",
    "PRISM 800m":    "#66C2A5",
    "Daymet-PC":     "#7570B3",
    "gridMET":       "#E7298A",
    "HRRR F01":      "#1F78B4",
    "HRRR F06":      "#A6CEE3",
    "UCLA ERA5 d02": "#E6AB02",
    "PNNL hist":     "#D95F02",
    "CONUS404":      "#666666",
}

# SNOTEL Station Colors (ColorBrewer Set1/Set2 inspired)
SNOTEL_COLORS = {
    "Beaver Pass":    "#e41a1c",
    "Marten Ridge":   "#377eb8",
    "Brown Top":      "#4daf4a",
    "Thunder Basin":  "#984ea3",
    "Swamp Creek":    "#ff7f00",
    "Decline Creek":  "#a65628",
    "Rainy Pass":     "#f781bf",
    "Easy Pass":      "#999999",
    "Hozomeen Camp":  "#4682B4",
}

def load_snotel_daily_precip(start, end):
    """Loads daily SNOTEL precipitation (converted to mm) for the event window."""
    zarr_file = os.path.join(SNOTEL_DIR, f"{start}_{end}_SNOTEL_daily_data.zarr")
    if not os.path.exists(zarr_file):
        print(f"[WARN] SNOTEL file not found: {zarr_file}")
        return None
    
    ds = xr.open_zarr(zarr_file)
    if "PRECIPITATION" not in ds:
        print(f"[WARN] PRECIPITATION variable not in {zarr_file}")
        ds.close()
        return None
    
    # SNOTEL precipitation is in inches, convert to mm (1 inch = 25.4 mm)
    da = ds["PRECIPITATION"] * 25.4
    da.attrs["units"] = "mm"
    
    # Store coordinates locally and close dataset
    da_loaded = da.load()
    ds.close()
    return da_loaded

def main():
    print("Loading Skagit Basin boundary geometry...")
    geom = lib.load_basin_geom()
    
    # Set up matplotlib style
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
    })
    
    fig, axes = plt.subplots(len(EVENTS), 2, figsize=(15, 22), facecolor='#ffffff')
    
    for idx, ev in enumerate(EVENTS):
        yr = ev["year"]
        start = ev["start"]
        end = ev["end"]
        label = ev["label"]
        
        print(f"\nProcessing Event {yr} ({start} to {end})...")
        
        # --- Column 1: Gridded Basin Average ---
        ax1 = axes[idx, 0]
        # Excluded ORNL from gridded series per request (since it is redundant/identical to Daymet)
        daily_series = [
            ("PRISM 4km",       lib.load_prism_daily(start, end, "4km", geom, "Basin", do_verify=False)),
            ("PRISM 800m",      lib.load_prism_daily(start, end, "800m", geom, "Basin", do_verify=False)),
            ("Daymet-PC",       lib.load_daymet_pc_daily(start, end, geom, "Basin", do_verify=False)),
            ("gridMET",         lib.load_gridmet_daily(yr, start, end, geom, "Basin", do_verify=False)),
            ("HRRR F01",        lib.load_hrrr_daily(start, end, geom, "Basin", do_verify=False, fh_hours=1)),
            ("HRRR F06",        lib.load_hrrr_daily(start, end, geom, "Basin", do_verify=False, fh_hours=6)),
            ("UCLA ERA5 d02",   lib.load_ucla_era5_d02_daily(start, end, geom, "Basin", do_verify=False)),
            ("PNNL hist",       lib.load_pnnl_daily(yr, start, end, geom, "Basin", do_verify=False)),
            ("CONUS404",        lib.load_conus_daily(start, end, geom, "Basin", do_verify=False)),
        ]
        
        max_y = 0.0
        
        for name, dly in daily_series:
            if lib.is_plottable_1d(dly):
                # Ensure the time index is sorted and unique
                dly = dly.sortby("time")
                cum = dly.cumsum("time")
                x = pd.to_datetime(cum["time"].values)
                y = np.asarray(cum.values, dtype=float)
                
                ax1.plot(x, y, label=name, color=GRID_COLORS.get(name, "#000000"), linewidth=2.0)
                max_y = max(max_y, float(y[-1]))
                print(f"  Gridded {name}: final cumulative average = {y[-1]:.2f} mm")
        
        ax1.set_title(f"{label} | Basin Average", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Cumulative Precip (mm)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(fontsize=8, loc='upper left')
        
        # --- Column 2: SNOTEL Station Data ---
        ax2 = axes[idx, 1]
        snotel_da = load_snotel_daily_precip(start, end)
        
        if snotel_da is not None:
            # Sort stations by elevation for better representation
            elevs = snotel_da.coords["elevation_ft"].values
            site_names = snotel_da.coords["site_name"].values
            sites = snotel_da.coords["site"].values
            
            sorted_indices = np.argsort(elevs)[::-1]
            
            for s_idx in sorted_indices:
                site_name = str(site_names[s_idx])
                site_id = str(sites[s_idx])
                elev = int(elevs[s_idx])
                
                # Extract station daily time series
                da_station = snotel_da.sel(site=site_id).sortby("time")
                
                # Fill NaN with 0 for cumulative sums
                da_station_filled = da_station.fillna(0)
                cum_station = da_station_filled.cumsum("time")
                
                x_st = pd.to_datetime(cum_station["time"].values)
                y_st = np.asarray(cum_station.values, dtype=float)
                
                label_str = f"{site_name} ({elev} ft)"
                ax2.plot(x_st, y_st, label=label_str, color=SNOTEL_COLORS.get(site_name, "#000000"), linewidth=2.0)
                max_y = max(max_y, float(y_st[-1]))
                print(f"  SNOTEL {site_name}: final cumulative precip = {y_st[-1]:.2f} mm")
        
        ax2.set_title(f"{label} | SNOTEL Stations", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Cumulative Precip (mm)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(fontsize=8, loc='upper left')
        
        # Apply identical y-axis limits to both columns in the same row
        ylim_max = max_y * 1.05
        ax1.set_ylim(0, ylim_max)
        ax2.set_ylim(0, ylim_max)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.35)
    
    plt.suptitle("Comparison of Cumulative Precipitation during Atmospheric River Events\n"
                 "Basin-Wide Gridded Average (Column 1) vs. Point SNOTEL Station Observations (Column 2)",
                 fontsize=16, fontweight='bold')
    
    out_file = os.path.join(OUT_DIR, "refined_basin_cumulative_and_stations_comparison.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\n[PLOT] Saved final comparison figure to: {out_file}")

if __name__ == "__main__":
    main()
