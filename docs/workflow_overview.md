# Workflow overview

This folder collects the most important pieces of the Skagit precipitation analysis in a simpler form.

## 1. Data preparation

The data preparation script reads the raw event-based precipitation dataset and creates a clean table with dates and precipitation totals.

## 2. Downloading inputs

If you need to fetch product data from the original sources, use the download wrappers in the scripts folder:

- download_prism.py
- download_hrrr.py
- download_gridmet.py
- download_conus.py
- download_snotel.py

These wrappers call the existing downloader scripts in the main scripts folder so the workflow stays centralized.

## 3. Preprocessing

The preprocessing script standardizes product columns and computes bias values relative to PRISM or Daymet.

## 4. Plotting

The plotting scripts generate the main figures used in the analysis.

- plot_prism_bias.py: PRISM-baseline manuscript-style figure
- plot_daymet_bias.py: Daymet-baseline appendix-style figure

## 5. Important notes

- HRRR is included but should be reviewed for basin integration.
- Sparse bins with very few values are shown as points rather than misleading boxplots.
- ORNL is currently excluded from the PRISM-focused manuscript figure.
