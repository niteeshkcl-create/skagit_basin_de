# Reproducible Skagit Precipitation Analysis

This folder is a simplified, beginner-friendly entry point for the Skagit precipitation workflow.

## What is included

- Data preparation and cleaning steps
- Preprocessing for basin-scale precipitation products
- Bias calculation against PRISM and Daymet baselines
- Plotting scripts for the manuscript-style figures
- Short notes explaining what each script does

## Folder structure

- scripts/: runnable Python scripts
- docs/: short explanations and workflow notes
- data/: expected input/output data locations
- notebooks/: optional notebooks for exploration

## Main workflow

1. Prepare the event list and bulk bias dataset
2. Download the required weather/product datasets if needed
3. Clean and standardize the precipitation products
4. Compute bias against the chosen baseline
5. Generate the manuscript-style bias plot

## Key output

- The main figure produced here is:
  - grouped_bias_3_metrics_2_periods.png

## Suggested order

1. Run any of the download wrappers as needed, such as download_prism.py, download_hrrr.py, download_gridmet.py, download_conus.py, download_snotel.py, download_backfill_daymet.py, download_backfill_gridmet.py, download_bulk_historical_snotel.py, download_ornl.py, download_wrf.py, or download_fetch_analysis_data.py
2. Run data_preparation.py
3. Run preprocess_products.py
4. Run plot_prism_bias.py
5. Run plot_daymet_bias.py
