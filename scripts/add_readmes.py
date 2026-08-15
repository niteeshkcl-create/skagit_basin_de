import os

base_path = "/data0/skagit_met/data_transfer/data/"
readmes = {
    "CONUS": "Continental US scale meteorological datasets used for broad spatial analysis.",
    "CanESM2": "CMIP5 Global Climate Model (GCM) outputs from the Canadian Earth System Model v2.",
    "DaymetV4": "High-resolution (1km) daily meteorological data from ORNL DAAC (Version 4).",
    "GFDL_ESM2M": "CMIP5 Global Climate Model (GCM) outputs from the Geophysical Fluid Dynamics Laboratory.",
    "HadGEM2_ES": "CMIP5 Global Climate Model (GCM) outputs from the Hadley Centre Global Environmental Model.",
    "MPI_ESM_MR": "CMIP5 Global Climate Model (GCM) outputs from the Max Planck Institute Earth System Model.",
    "NLDAS": "North American Land Data Assimilation System (NLDAS) hourly/daily forcing data.",
    "PNNL": "6-km resolution WRF-downscaled regional climate simulations (SERDP project) from PNNL.",
    "PRISM": "Organized Parameter-elevation Regressions on Independent Slopes Model (PRISM) Zarr archives (1981-Present).",
    "atmospheric_rivers": "Event-based meteorological subsets clipped for specific Atmospheric River events.",
    "cache": "Temporary computational caches, kerchunk indices, and intermediate analysis artifacts.",
    "climate_sets": "Continuous multi-decadal meteorological stacks (e.g., ORNL 1981-2011) used for baseline bias analysis.",
    "daymet_nc_2014_2024": "Raw NetCDF fragments of Daymet data for the 2014-2024 monitoring period.",
    "daymet_subsets": "Spatial or temporal sub-selections of Daymet data for specialized local studies.",
    "derived": "Calculated datasets derived from raw meteorological forcing (e.g., indices, anomalies).",
    "gridmet": "Gridded Surface Meteorological (gridMET) datasets, often used as secondary observations.",
    "ornl": "Metadata and documentation for datasets sourced from Oak Ridge National Laboratory.",
    "ornl_nc": "Raw NetCDF versions of ORNL/Daymet data releases.",
    "prism": "Legacy or non-organized raw NetCDF/Zip fragments of PRISM meteorological data.",
    "prism_nc": "Raw PRISM NetCDF files directly from the PRISM climate group.",
    "prism_ppt": "PRISM precipitation (ppt) specific raw subsets and legacy fetches.",
    "prism_ppt_800m": "High-resolution 800m PRISM precipitation data for fine-scale hydrologic analysis.",
    "prism_temp_nc": "PRISM temperature variable (tmax/tmin) specific NetCDF files.",
    "snotel": "Snow Telemetry (SNOTEL) station observation data and automated Zarr archives.",
    "ucla_era5_d02_daily": "ERA5 boundary conditions and forcing files used for UCLA WRF Domain 02 simulations.",
    "ucla_wrf_outputs": "Raw and post-processed outputs from the UCLA Weather Research and Forecasting (WRF) model.",
    "ucla_wrf_outputs_new": "Updated or experimental UCLA WRF simulation results.",
    "weather_data": "Assorted meteorological forcing files and testing subsets.",
    "weather_data_06": "Weather data subsets likely relating to the 2006 flood event or specific simulation runs."
}

for folder, description in readmes.items():
    folder_path = os.path.join(base_path, folder)
    if os.path.isdir(folder_path):
        readme_path = os.path.join(folder_path, "README.md")
        content = f"# {folder}\n\n{description}\n"
        try:
            with open(readme_path, "w") as f:
                f.write(content)
            print(f"✓ Created README in {folder}")
        except PermissionError:
            print(f"✗ Permission denied in {folder}")
        except Exception as e:
            print(f"! Error in {folder}: {e}")
    else:
        print(f"? Folder {folder} not found")
