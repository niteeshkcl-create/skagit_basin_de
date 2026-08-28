import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
prism_path = "/data0/skagit_met/data_transfer/data/PRISM/2001-2010/2003-01-01_2003-12-31_daily_4km_PRISM_data.zarr"
huc8_geo = "/data0/hernanqd/plots_code/skagit_basin_de/data/GIS/SkagitSubBasin_HUC8.geojson"

print("Loading PRISM 4km dataset...")
ds = xr.open_zarr(prism_path, consolidated=False)
prism_data = ds['ppt'].isel(time=0).values  # First day for visualization

print("Loading Skagit Basin geometry...")
gdf = gpd.read_file(huc8_geo).to_crs("EPSG:4326")

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Plot PRISM grid - show data availability
im = ax.imshow(prism_data, extent=[ds.lon.min().values, ds.lon.max().values,
                                    ds.lat.min().values, ds.lat.max().values],
               origin='lower', cmap='YlGnBu', alpha=0.6, label='PRISM 4km Grid')

# Plot basin boundary
gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2.5, label='Skagit Basin (HUC8)')

# Format
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('PRISM 4km Dataset Spatial Extent vs Skagit Basin', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, label='Precipitation (mm) - Jan 1, 2003')

plt.tight_layout()
output_path = "/data0/hernanqd/plots_code/prism_4km_spatial_extent.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved to: {output_path}")

# Print info
print(f"\nPRISM 4km Dataset Info:")
print(f"  Lon range: {ds.lon.min().values:.3f} to {ds.lon.max().values:.3f}")
print(f"  Lat range: {ds.lat.min().values:.3f} to {ds.lat.max().values:.3f}")
print(f"  Grid cells: {len(ds.lon)} × {len(ds.lat)}")
print(f"\nSkagit Basin bounds:")
print(f"  Lon range: {gdf.total_bounds[0]:.3f} to {gdf.total_bounds[2]:.3f}")
print(f"  Lat range: {gdf.total_bounds[1]:.3f} to {gdf.total_bounds[3]:.3f}")

ds.close()
