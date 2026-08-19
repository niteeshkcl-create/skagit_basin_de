"""
plot_ar_event_spatial.py
------------------------
For each of the 5 AR events (2003-Oct, 2006-Nov, 2016-Oct, 2018-Dec, 2021-Nov),
generate a spatial multi-product precipitation map matching the style of
  may 23/spatial_seasonal_avg_products.png
but averaged over the days of each AR event window.

Output: jun 8/<event_label>_spatial_precip.png  (one file per event)
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
from scipy.interpolate import griddata

warnings.filterwarnings('ignore')

# --- Configuration & Paths ---
BASE_DIR = "/data0/nksp2/skagit/skagit_2/skagit-met"
VAULT_DIR = "/data0/skagit_met/data_transfer/data"
BOUNDARY_PATH = os.path.join(BASE_DIR, "data/GIS/SkagitBoundary.json")
SUBBASIN_PATH  = os.path.join(BASE_DIR, "data/GIS/SkagitSubBasin_HUC8.geojson")
OUT_DIR = os.path.join("/data0/hernanqd/plots_code/skagit_basin_de/spatial_plots/plots/plot_ar_event_spatial")
os.makedirs(OUT_DIR, exist_ok=True)

# --- AR Event windows (start, end inclusive) ---
# AR_EVENTS = [
#     {"label": "October_2003_AR5",   "start": "2003-10-18", "end": "2003-10-24", "row_label": "Oct 18–24, 2003\n(AR5)"},
#     {"label": "November_2006_AR4",  "start": "2006-11-04", "end": "2006-11-10", "row_label": "Nov 4–10, 2006\n(AR4)"},
#     {"label": "October_2016_AR3",   "start": "2016-10-30", "end": "2016-11-05", "row_label": "Oct 30–Nov 5, 2016\n(AR3)"},
#     {"label": "December_2018_AR2",  "start": "2018-12-15", "end": "2018-12-21", "row_label": "Dec 15–21, 2018\n(AR2)"},
#     {"label": "November_2021_AR5",  "start": "2021-11-10", "end": "2021-11-17", "row_label": "Nov 10–17, 2021\n(AR5)"},
# ]

AR_EVENTS = [
    {"label": "November_1990_AR5",   "start": "1990-11-23", "end": "1990-11-30", "row_label": "Nov 23–30, 1990\n(AR5)"},
    {"label": "November_2024_AR5",  "start": "2024-11-22", "end": "2024-11-29", "row_label": "Nov 22–29, 2024\n(AR5)"},
    {"label": "November_2021_AR4",   "start": "2021-11-13", "end": "2021-11-20", "row_label": "Nov 13–20, 2021\n(AR4)"},
    {"label": "December_2010_AR3",  "start": "2010-12-11", "end": "2010-12-18", "row_label": "Dec 11–18, 2010\n(AR3)"},
    {"label": "January_1984_AR4",  "start": "1984-01-03", "end": "1984-01-10", "row_label": "Jan 3–10, 1984\n(AR4)"},
]

# Products to plot (same order as the seasonal figure)
PRODUCTS = ['PRISM', 'Daymet', 'ORNL (Daymet)', 'PNNL', 'CONUS404', 'UCLA', 'GridMET', 'HRRR']

# --- Load static coordinates once ---
print("Loading static grid coordinates...")

ds_pnnl_static = xr.open_dataset(os.path.join(VAULT_DIR, "PNNL/SERDP6km.geo_em.d01.nc"))
pnnl_lon_full = ds_pnnl_static.XLONG_M.values[0]
pnnl_lat_full = ds_pnnl_static.XLAT_M.values[0]
ds_pnnl_static.close()

ds_ucla_static = xr.open_dataset(os.path.join(VAULT_DIR, "ucla_era5_d02_daily/static/wrfinput_d02_coord.nc"))
ucla_lon = ds_ucla_static.lon2d.values
ucla_lat = ds_ucla_static.lat2d.values
ds_ucla_static.close()

# HRRR coordinates
hrrr_lon = None
hrrr_lat = None
for y in range(2014, 2022):
    for m in range(1, 13):
        p = os.path.join(VAULT_DIR, "weather_data", f"{y}-{m:02d}_HRRR_data.zarr")
        if os.path.exists(p):
            try:
                ds_h = xr.open_zarr(p, consolidated=False)
                hrrr_lon = ds_h.longitude.values
                hrrr_lat = ds_h.latitude.values
                ds_h.close()
                break
            except Exception:
                pass
    if hrrr_lon is not None:
        break

# Crop bounding box: Skagit domain
BB = (-122.5, -120.5, 47.8, 49.5)  # lon_min, lon_max, lat_min, lat_max
MAP_EXTENT = [-122.35, -120.65, 47.90, 49.35]

def bb_mask(lon2d, lat2d):
    return (lon2d >= BB[0]) & (lon2d <= BB[1]) & (lat2d >= BB[2]) & (lat2d <= BB[3])

def crop_rows_cols(lon2d, lat2d):
    m = bb_mask(lon2d, lat2d)
    rows, cols = np.where(m)
    return rows.min(), rows.max(), cols.min(), cols.max()

# Crop reference PNNL 6km grid
row_min_p, row_max_p, col_min_p, col_max_p = crop_rows_cols(pnnl_lon_full, pnnl_lat_full)
ref_lon = pnnl_lon_full[row_min_p:row_max_p+1, col_min_p:col_max_p+1]
ref_lat = pnnl_lat_full[row_min_p:row_max_p+1, col_min_p:col_max_p+1]
print(f"Reference 6km grid shape: {ref_lon.shape}")

# Crop UCLA & HRRR coordinate arrays
row_min_u, row_max_u, col_min_u, col_max_u = crop_rows_cols(ucla_lon, ucla_lat)
ucla_lon_c = ucla_lon[row_min_u:row_max_u+1, col_min_u:col_max_u+1]
ucla_lat_c = ucla_lat[row_min_u:row_max_u+1, col_min_u:col_max_u+1]

if hrrr_lon is not None:
    row_min_h, row_max_h, col_min_h, col_max_h = crop_rows_cols(hrrr_lon, hrrr_lat)
    hrrr_lon_c = hrrr_lon[row_min_h:row_max_h+1, col_min_h:col_max_h+1]
    hrrr_lat_c = hrrr_lat[row_min_h:row_max_h+1, col_min_h:col_max_h+1]
else:
    hrrr_lon_c = hrrr_lat_c = None

# Load GIS boundaries
boundary_gdf = gpd.read_file(BOUNDARY_PATH)
subbasin_gdf = gpd.read_file(SUBBASIN_PATH) if os.path.exists(SUBBASIN_PATH) else None

# Precompute watershed mask on the reference 6km grid
if hasattr(boundary_gdf.to_crs("EPSG:4326"), 'union_all'):
    poly = boundary_gdf.to_crs("EPSG:4326").union_all()
else:
    poly = boundary_gdf.to_crs("EPSG:4326").unary_union

df_pts = pd.DataFrame({'lon': ref_lon.flatten(), 'lat': ref_lat.flatten()})
gdf_pts = gpd.GeoDataFrame(df_pts, geometry=gpd.points_from_xy(df_pts.lon, df_pts.lat), crs="EPSG:4326")
inside = gdf_pts.intersects(poly).values
mask_2d = inside.reshape(ref_lon.shape)


def regrid_to_6km(da_src, mask_2d=None):
    """Regrid any product to the PNNL 6km reference grid via griddata."""
    if da_src is None:
        return np.full(ref_lat.shape, np.nan)
    vals = da_src.values
    if np.all(np.isnan(vals)):
        return np.full(ref_lat.shape, np.nan)

    lon_name = next((c for c in da_src.coords if 'lon' in c or 'longitude' in c), None)
    lat_name = next((c for c in da_src.coords if 'lat' in c or 'latitude' in c), None)
    if lon_name is None or lat_name is None:
        return np.full(ref_lat.shape, np.nan)

    lon_vals = da_src.coords[lon_name].values
    lat_vals = da_src.coords[lat_name].values

    if lon_vals.ndim == 1 and lat_vals.ndim == 1:
        lon_g, lat_g = np.meshgrid(lon_vals, lat_vals)
    else:
        lon_g, lat_g = lon_vals, lat_vals

    pts = np.column_stack((lon_g.flatten(), lat_g.flatten()))
    v   = vals.flatten()
    ok  = ~np.isnan(v)
    if not ok.any():
        return np.full(ref_lat.shape, np.nan)

    grid_z = griddata(pts[ok], v[ok], (ref_lon, ref_lat), method='linear')
    if mask_2d is not None:
        grid_z[~mask_2d] = np.nan
    return grid_z


def get_prism_zarr(year):
    def decade(y):
        s = (y // 10) * 10
        if s == 1980 and y > 1980: return "1981-1990"
        if y % 10 == 0 and y > 1980: return f"{s-9}-{s}"
        if s % 10 == 0: return f"{s+1}-{s+10}"
        return f"{s}-{(y//10)*10+9}"
    dec = decade(year)
    p = os.path.join(VAULT_DIR, "PRISM", dec, f"{year}-01-01_{year}-12-31_daily_4km_PRISM_data.zarr")
    return p if os.path.exists(p) else None

def get_ornl_zarr(year):
    for p in [
        os.path.join(VAULT_DIR, "climate_sets/1981_2011_ORNL_data.zarr") if 1981 <= year <= 2011 else None,
        os.path.join(VAULT_DIR, f"ornl_nc/{year}_{year}_ref_DaymetV4_ORNL_data.zarr"),
        os.path.join(VAULT_DIR, f"ornl/{year}_{year}_ref_DaymetV4_ORNL_data.zarr"),
        os.path.join(VAULT_DIR, f"DaymetV4/{year}_{year}_ORNL_data.zarr"),
    ]:
        if p and os.path.exists(p): return p
    return None


def load_event_grids(event):
    """Load daily-mean precipitation from each product for the event window."""
    start = event["start"]
    end   = event["end"]
    year  = int(start[:4])
    print(f"\n  Loading data for {event['label']} ({start} → {end})...")

    grids = {p: None for p in PRODUCTS}

    # 1. PRISM
    try:
        pz = get_prism_zarr(year)
        if pz:
            ds = xr.open_zarr(pz, consolidated=False)
            da = ds['ppt'].sel(time=slice(start, end)).mean(dim='time').compute()
            grids['PRISM'] = da
            ds.close()
            print("    PRISM OK")
    except Exception as e:
        print(f"    [WARN] PRISM: {e}")

    # 2. Daymet & ORNL (same source)
    try:
        dz = get_ornl_zarr(year)
        if dz:
            is_cons = (1981 <= year <= 2011)
            ds = xr.open_zarr(dz, consolidated=is_cons)
            if 'day' in ds.dims:
                ds = ds.rename({'day': 'time'})
            if 'lat' in ds.dims and 'lon' in ds.dims:
                if len(ds.lat) == 66 and len(ds.lon) == 78:
                    ds = ds.isel(lat=slice(None, None, 2), lon=slice(None, None, 2))
            var = 'prcp' if 'prcp' in ds.data_vars else 'ppt'
            da = ds[var].sel(time=slice(start, end)).mean(dim='time').compute()
            grids['Daymet']       = da
            grids['ORNL (Daymet)'] = da
            if not is_cons: ds.close()
            print("    Daymet/ORNL OK")
    except Exception as e:
        print(f"    [WARN] Daymet/ORNL: {e}")

    # 3. PNNL WRF
    try:
        pnnl_file = os.path.join(VAULT_DIR, "PNNL/historical", str(year),
                                 f"PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{year}.nc")
        if os.path.exists(pnnl_file):
            ds = xr.open_dataset(pnnl_file, chunks={'time': 720})
            da_daily = ds['PREC_ACC_NC'].resample(time='1D').sum()
            # Crop
            mask_c = bb_mask(pnnl_lon_full, pnnl_lat_full)
            rows, cols = np.where(mask_c)
            rm, rx, cm, cx = rows.min(), rows.max(), cols.min(), cols.max()
            da_daily = da_daily.isel(x=slice(rm, rx+1), y=slice(cm, cx+1))
            lat_c = pnnl_lat_full[rm:rx+1, cm:cx+1]
            lon_c = pnnl_lon_full[rm:rx+1, cm:cx+1]
            da = da_daily.sel(time=slice(start, end)).mean(dim='time').compute()
            da = da.assign_coords(lat=(('x', 'y'), lat_c), lon=(('x', 'y'), lon_c))
            grids['PNNL'] = da
            ds.close()
            print("    PNNL OK")
    except Exception as e:
        print(f"    [WARN] PNNL: {e}")

    # 4. CONUS404
    try:
        conus_path = os.path.join(BASE_DIR, "data/weather_data/conus404_skagit_precip_daily_full.zarr")
        ds = xr.open_zarr(conus_path)
        mask_c = bb_mask(ds.lon.values, ds.lat.values)
        rows, cols = np.where(mask_c)
        if len(rows):
            rm, rx, cm, cx = rows.min(), rows.max(), cols.min(), cols.max()
            ds = ds.isel(y=slice(rm, rx+1), x=slice(cm, cx+1))
        da = ds['precip_daily'].sel(time=slice(start, end)).mean(dim='time').compute()
        grids['CONUS404'] = da
        ds.close()
        print("    CONUS404 OK")
    except Exception as e:
        print(f"    [WARN] CONUS404: {e}")

    # 5. UCLA ERA5 d02
    try:
        da_parts = []
        for yr_off in [year-1, year]:
            p = os.path.join(VAULT_DIR, "ucla_era5_d02_daily", "prec",
                             f"prec.daily.era5.d02.{yr_off}.nc")
            if os.path.exists(p):
                file_start = f"{yr_off}-09-01"
                file_end = f"{yr_off+1}-08-31"
                s_start = max(start, file_start)
                s_end = min(end, file_end)
                if s_start <= s_end:
                    ds_u = xr.open_dataset(p)
                    u_var = "prec" if "prec" in ds_u.data_vars else "pr"
                    da_part = ds_u[u_var].sel(day=slice(s_start, s_end)).compute()
                    da_parts.append(da_part)
                    ds_u.close()
        if da_parts:
            da_year = xr.concat(da_parts, dim='day')
            da_year = da_year.isel(lat2d=slice(row_min_u, row_max_u+1),
                                   lon2d=slice(col_min_u, col_max_u+1))
            da = da_year.mean(dim='day')
            da = da.assign_coords(lat=(('lat2d', 'lon2d'), ucla_lat_c),
                                  lon=(('lat2d', 'lon2d'), ucla_lon_c))
            grids['UCLA'] = da
            print("    UCLA OK")
    except Exception as e:
        print(f"    [WARN] UCLA: {e}")

    # 6. GridMET
    try:
        gridmet_path = os.path.join(VAULT_DIR, "gridmet", f"{year}_daily_4km_gridMET_data.zarr")
        if os.path.exists(gridmet_path):
            ds = xr.open_zarr(gridmet_path)
            if 'day' in ds.dims:
                ds = ds.rename({'day': 'time'})
            da = ds['prcp'].sel(time=slice(start, end)).mean(dim='time').compute()
            grids['GridMET'] = da
            ds.close()
            print("    GridMET OK")
    except Exception as e:
        print(f"    [WARN] GridMET: {e}")

    # 7. HRRR  (available 2014–)
    if year >= 2014 and hrrr_lon_c is not None:
        try:
            start_dt = pd.Timestamp(start)
            end_dt   = pd.Timestamp(end)
            months_needed = set()
            cur = start_dt
            while cur <= end_dt:
                months_needed.add((cur.year, cur.month))
                cur += pd.DateOffset(months=1)
            # Also include the next month in case event spans a boundary
            months_needed.add(((end_dt + pd.DateOffset(months=1)).year,
                                (end_dt + pd.DateOffset(months=1)).month))

            da_months = []
            for (yr_m, mo_m) in sorted(months_needed):
                for suffix in ["", "_fixed_time"]:
                    p = os.path.join(VAULT_DIR, "weather_data",
                                     f"{yr_m}-{mo_m:02d}_HRRR_data{suffix}.zarr")
                    if os.path.exists(p):
                        ds_h = xr.open_zarr(p, consolidated=False)
                        ds_h = ds_h.isel(y=slice(row_min_h, row_max_h+1),
                                         x=slice(col_min_h, col_max_h+1))
                        if "forecast_hour" in ds_h.coords:
                            ds_h = ds_h.where(ds_h.forecast_hour == 1, drop=True)
                        elif "step" in ds_h.coords:
                            if np.issubdtype(ds_h.step.dtype, np.timedelta64):
                                ds_h = ds_h.where(ds_h.step == np.timedelta64(1, "h"), drop=True)
                            else:
                                ds_h = ds_h.where(ds_h.step == 1, drop=True)
                        ds_h = ds_h.sortby('time').drop_duplicates('time')
                        v_h = 'APCP_sfc' if 'APCP_sfc' in ds_h.data_vars else 'tp'
                        da_d = ds_h[v_h].resample(time='1D').sum().compute()
                        da_d = da_d.drop_vars(['latitude', 'longitude'], errors='ignore')
                        da_months.append(da_d)
                        ds_h.close()
                        break   # prefer non-fixed unless only fixed exists

            if da_months:
                da_all = xr.concat(da_months, dim='time').sortby('time').drop_duplicates('time')
                da = da_all.sel(time=slice(start, end)).mean(dim='time')
                da = da.assign_coords(lat=(('y', 'x'), hrrr_lat_c),
                                      lon=(('y', 'x'), hrrr_lon_c))
                grids['HRRR'] = da
                print("    HRRR OK")
        except Exception as e:
            print(f"    [WARN] HRRR: {e}")
    else:
        print("    HRRR skipped (pre-2014 event or no grid)")

    return grids


def plot_event(event, grids):
    """Plot the 1×8 product spatial map for a single AR event."""
    label   = event["label"]
    start   = event["start"]
    end     = event["end"]
    out_png = os.path.join(OUT_DIR, f"{label}_spatial_precip.png")

    # Regrid all products to 6km reference
    print(f"  Regridding {label}...")
    rg = {}
    for prod in PRODUCTS:
        rg[prod] = regrid_to_6km(grids[prod], mask_2d)

    # Dynamic vmax: 99.5th pct across all product grids
    all_vals = []
    for prod in PRODUCTS:
        g = rg[prod]
        if g is not None:
            v = g[~np.isnan(g)]
            all_vals.extend(v)
    all_vals = np.array(all_vals)
    if len(all_vals):
        vmax = max(float(np.ceil(np.percentile(all_vals, 99.5))), 1.0)
    else:
        vmax = 20.0
    vmin = 0.0

    print(f"  Color range: {vmin}–{vmax} mm/day")

    # Figure: 1 row × 8 cols
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
    })

    fig, axes = plt.subplots(
        1, len(PRODUCTS),
        figsize=(28, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
        facecolor='#ffffff'
    )

    im = None
    for col_idx, prod in enumerate(PRODUCTS):
        ax = axes[col_idx]
        grid = rg[prod]

        if grid is not None and not np.all(np.isnan(grid)):
            im = ax.pcolormesh(
                ref_lon, ref_lat, grid,
                transform=ccrs.PlateCarree(),
                cmap="Greens", vmin=vmin, vmax=vmax,
                shading='auto'
            )
        else:
            # Grey placeholder if product missing
            ax.set_facecolor('#dddddd')
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='#555555')

        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6, edgecolor='#444444')
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.8, edgecolor='#000000')
        ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle='--', linewidth=0.4, edgecolor='#666666')

        if subbasin_gdf is not None:
            subbasin_gdf.plot(ax=ax, facecolor="none", edgecolor="#888888",
                              lw=0.5, linestyle=":", transform=ccrs.PlateCarree())
        boundary_gdf.plot(ax=ax, facecolor="none", edgecolor="#000000",
                          lw=1.3, transform=ccrs.PlateCarree())

        # Column title (product name + year range)
        hrrr_note = "\n(2014–)" if prod == "HRRR" else ""
        ax.set_title(f"{prod}{hrrr_note}", fontsize=11, fontweight='bold', pad=8)

    # Colorbar
    if im is not None:
        cbar_ax = fig.add_axes([0.15, -0.04, 0.70, 0.03])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                            label="Average Daily Precipitation (mm/day)")
        cbar.ax.tick_params(labelsize=11)

    # Overall title
    event_name = label.replace("_", " ")
    plt.suptitle(
        f"Spatial Distribution of Precipitation — {event_name}\n"
        f"Multi-Product Comparison ({start} to {end})",
        fontsize=15, fontweight='bold', y=1.04
    )

    plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.05, wspace=0.04)
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_png}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nOutput directory: {OUT_DIR}\n")
    for event in AR_EVENTS:
        print(f"\n{'='*60}")
        print(f"Processing: {event['label']}")
        print(f"{'='*60}")
        grids = load_event_grids(event)
        plot_event(event, grids)

    print("\nAll done!")
