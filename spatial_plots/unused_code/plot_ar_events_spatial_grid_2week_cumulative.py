"""
plot_ar_events_spatial_grid_2week_cumulative.py
----------------------------------------------
Generates a single 5 rows x 8 columns grid of spatial multi-product precipitation maps,
where each row corresponds to one of the 5 AR events and each column is a dataset.
Each cell displays the CUMULATIVE precipitation at the end of the 2-week event period.

Output: jun 8/ar_events_spatial_comparison_2week_cumulative.png
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
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
OUT_DIR = os.path.join("/data0/hernanqd/plots_code/spatial_plots/plots/plot_ar_events_spatial_grid_2week_cumulative")
os.makedirs(OUT_DIR, exist_ok=True)

# --- AR Event 2-week windows ---
AR_EVENTS = [
    {"label": "October_2003_AR5",   "start": "2003-10-18", "end": "2003-10-31", "row_label": "Oct 18–31, 2003\n(AR5)"},
    {"label": "November_2006_AR4",  "start": "2006-11-04", "end": "2006-11-17", "row_label": "Nov 4–17, 2006\n(AR4)"},
    {"label": "October_2016_AR3",   "start": "2016-10-30", "end": "2016-11-12", "row_label": "Oct 30–Nov 12, 2016\n(AR3)"},
    {"label": "December_2018_AR2",  "start": "2018-12-15", "end": "2018-12-28", "row_label": "Dec 15–28, 2018\n(AR2)"},
    {"label": "November_2021_AR5",  "start": "2021-11-10", "end": "2021-11-23", "row_label": "Nov 10–23, 2021\n(AR5)"},
]

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
    """Load CUMULATIVE precipitation from each product for the 2-week event window."""
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
            da = ds['ppt'].sel(time=slice(start, end)).sum(dim='time').compute()
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
            da = ds[var].sel(time=slice(start, end)).sum(dim='time').compute()
            grids['Daymet']       = da
            grids['ORNL (Daymet)'] = da
            if not is_cons: ds.close()
            print("    Daymet/ORNL OK")
    except Exception as e:
        print(f"    [WARN] Daymet/ORNL: {e}")

    # 3. PNNL WRF (hourly accumulated → daily diff → event sum)
    try:
        pnnl_file = os.path.join(VAULT_DIR, "PNNL/historical", str(year),
                                 f"PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{year}.nc")
        if os.path.exists(pnnl_file):
            ds = xr.open_dataset(pnnl_file, chunks={'time': 720})
            da_slice = ds['PREC_ACC_NC'].sel(time=slice(start, end))
            diff = da_slice.diff('time')
            curr = da_slice.isel(time=slice(1, None))
            inc = xr.where(diff >= 0, diff, curr)
            da = inc.sum(dim='time').compute()
            
            # Crop
            mask_c = bb_mask(pnnl_lon_full, pnnl_lat_full)
            rows, cols = np.where(mask_c)
            rm, rx, cm, cx = rows.min(), rows.max(), cols.min(), cols.max()
            da = da.isel(x=slice(rm, rx+1), y=slice(cm, cx+1))
            lat_c = pnnl_lat_full[rm:rx+1, cm:cx+1]
            lon_c = pnnl_lon_full[rm:rx+1, cm:cx+1]
            da = da.assign_coords(lat=(('x', 'y'), lat_c), lon=(('x', 'y'), lon_c))
            grids['PNNL'] = da
            ds.close()
            print("    PNNL OK")
    except Exception as e:
        print(f"    [WARN] PNNL: {e}")

    # 4. CONUS404 (daily totals)
    try:
        conus_path = os.path.join(BASE_DIR, "data/weather_data/conus404_skagit_precip_daily_full.zarr")
        ds = xr.open_zarr(conus_path)
        mask_c = bb_mask(ds.lon.values, ds.lat.values)
        rows, cols = np.where(mask_c)
        if len(rows):
            rm, rx, cm, cx = rows.min(), rows.max(), cols.min(), cols.max()
            ds = ds.isel(y=slice(rm, rx+1), x=slice(cm, cx+1))
        da = ds['precip_daily'].sel(time=slice(start, end)).sum(dim='time').compute()
        grids['CONUS404'] = da
        ds.close()
        print("    CONUS404 OK")
    except Exception as e:
        print(f"    [WARN] CONUS404: {e}")

    # 5. UCLA ERA5 d02 (daily totals)
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
            da = da_year.sum(dim='day')
            da = da.assign_coords(lat=(('lat2d', 'lon2d'), ucla_lat_c),
                                  lon=(('lat2d', 'lon2d'), ucla_lon_c))
            grids['UCLA'] = da
            print("    UCLA OK")
    except Exception as e:
        print(f"    [WARN] UCLA: {e}")

    # 6. GridMET (daily totals)
    try:
        gridmet_path = os.path.join(VAULT_DIR, "gridmet", f"{year}_daily_4km_gridMET_data.zarr")
        if os.path.exists(gridmet_path):
            ds = xr.open_zarr(gridmet_path)
            if 'day' in ds.dims:
                ds = ds.rename({'day': 'time'})
            da = ds['prcp'].sel(time=slice(start, end)).sum(dim='time').compute()
            grids['GridMET'] = da
            ds.close()
            print("    GridMET OK")
    except Exception as e:
        print(f"    [WARN] GridMET: {e}")

    # 7. HRRR (available 2014–, hourly forecast steps)
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
                        
                        # Hourly diffs/accum conversion to daily sum
                        da_d = ds_h[v_h].resample(time='1D').sum().compute()
                        da_d = da_d.drop_vars(['latitude', 'longitude'], errors='ignore')
                        da_months.append(da_d)
                        ds_h.close()
                        break

            if da_months:
                da_all = xr.concat(da_months, dim='time').sortby('time').drop_duplicates('time')
                da = da_all.sel(time=slice(start, end)).sum(dim='time')
                da = da.assign_coords(lat=(('y', 'x'), hrrr_lat_c),
                                      lon=(('y', 'x'), hrrr_lon_c))
                grids['HRRR'] = da
                print("    HRRR OK")
        except Exception as e:
            print(f"    [WARN] HRRR: {e}")
    else:
        print("    HRRR skipped (pre-2014 event or no grid)")

    return grids


def main():
    # Load and regrid all events
    event_regridded_grids = {}
    
    for event in AR_EVENTS:
        print(f"\nProcessing: {event['label']}")
        grids = load_event_grids(event)
        
        # Regrid each product
        event_regridded_grids[event['label']] = {}
        for prod in PRODUCTS:
            print(f"  Regridding {prod} for {event['label']}...")
            event_regridded_grids[event['label']][prod] = regrid_to_6km(grids[prod], mask_2d)

    # Dynamic vmax: 99.5th percentile across all non-NaN values in all products/events
    print("\nCalculating dynamic colorbar limits...")
    all_vals = []
    for elabel in event_regridded_grids:
        for prod in PRODUCTS:
            g = event_regridded_grids[elabel][prod]
            if g is not None:
                v = g[~np.isnan(g)]
                all_vals.extend(v)
    all_vals = np.array(all_vals)
    if len(all_vals):
        vmax = max(float(np.ceil(np.percentile(all_vals, 99.5))), 1.0)
    else:
        vmax = 600.0
    vmin = 0.0
    print(f"Global color range: {vmin}–{vmax} mm")

    # Plot grid: 5 rows × 8 columns
    plt.rcParams.update({
        'font.size': 13,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
    })

    fig, axes = plt.subplots(
        len(AR_EVENTS), len(PRODUCTS),
        figsize=(25, 17.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
        facecolor='#ffffff'
    )

    im = None
    for row_idx, event in enumerate(AR_EVENTS):
        elabel = event['label']
        for col_idx, prod in enumerate(PRODUCTS):
            ax = axes[row_idx, col_idx]
            grid = event_regridded_grids[elabel][prod]

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
                        ha='center', va='center', fontsize=12, color='#555555')

            ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6, edgecolor='#444444')
            ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.8, edgecolor='#000000')
            ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle='--', linewidth=0.4, edgecolor='#666666')

            if subbasin_gdf is not None:
                subbasin_gdf.plot(ax=ax, facecolor="none", edgecolor="#888888",
                                  lw=0.5, linestyle=":", transform=ccrs.PlateCarree())
            boundary_gdf.plot(ax=ax, facecolor="none", edgecolor="#000000",
                              lw=1.3, transform=ccrs.PlateCarree())

            # Row label on the first column
            if col_idx == 0:
                ax.text(-0.25, 0.5, event['row_label'], transform=ax.transAxes,
                        fontsize=16, fontweight='bold', va='center', ha='right')

        # Column titles on the first row
        if row_idx == 0:
            for col_idx, prod in enumerate(PRODUCTS):
                label = "ORNL (Daymet)" if prod == "ORNL" else prod
                axes[0, col_idx].set_title(label, fontsize=16, fontweight='bold', pad=12)

    # Colorbar at the bottom of the grid
    if im is not None:
        cbar_ax = fig.add_axes([0.30, 0.05, 0.40, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                            label="Cumulative Precipitation (mm)")
        cbar.ax.tick_params(labelsize=13)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.10, wspace=0.05, hspace=0.08)

    # Overall title
    plt.suptitle(
        "Spatial Distribution of Precipitation during Atmospheric River Events\n"
        "Multi-Product Comparison (2-Week Cumulative Precipitation)",
        fontsize=22, fontweight='bold', y=0.96
    )

    out_png = os.path.join(OUT_DIR, "ar_events_spatial_comparison_2week_cumulative.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved combined comparison figure to: {out_png}")


if __name__ == "__main__":
    main()
