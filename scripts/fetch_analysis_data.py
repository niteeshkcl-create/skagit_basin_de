import os
import requests
import pandas as pd
from datetime import datetime

def download_usgs_data(site_id, start_date, end_date, output_file):
    # parameterCd=00060 (Discharge), statCd=00003 (Mean)
    url = f"https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={site_id}&startDT={start_date}&endDT={end_date}&parameterCd=00060&statCd=00003"
    print(f"Downloading USGS data from: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_file, 'w') as f:
            f.write(response.text)
        print(f"USGS data saved to {output_file}")
    else:
        print(f"Failed to download USGS data: {response.status_code}")

def download_ar_catalog(lat, lon, output_file):
    # Guessing the URL pattern for CW3E AR Catalog (Landfalling)
    # The interactive map points to specific files. 
    # For 48.25N, 122.5W, let's try to find the closest point.
    # Often it's just the lat/lon in the filename.
    # URL pattern from research: https://cw3e.ucsd.edu/cw3e-landfalling-ar-catalog/ (interactive)
    # Let's try to find the actual CSV link if possible.
    # Based on similar tasks, it might be:
    # https://cw3e.ucsd.edu/Projects/ARCatalog/ERA5/AR_Catalog_ERA5_48.0N_122.5W.csv
    
    # Actually, the user provided: https://cw3e.ucsd.edu/Projects/ARCatalog/catalog.html
    # This page has a "Download Catalog" button.
    # The full catalog is often available as a NetCDF or CSV.
    
    url = "https://cw3e.ucsd.edu/Projects/ARCatalog/ERA5/AR_Catalog_ERA5_48.25N_122.5W.csv"
    print(f"Attempting to download AR catalog from: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_file, 'w') as f:
            f.write(response.text)
        print(f"AR catalog saved to {output_file}")
    else:
        print(f"Failed to download AR catalog from guessed URL: {response.status_code}")
        # Try another common pattern
        url2 = "https://cw3e.ucsd.edu/Projects/ARCatalog/ERA5/AR_Catalog_ERA5_48.5N_122.5W.csv"
        print(f"Attempting alternative: {url2}")
        response = requests.get(url2)
        if response.status_code == 200:
            with open(output_file, 'w') as f:
                f.write(response.text)
            print(f"AR catalog saved to {output_file}")
        else:
            print(f"Failed alternative download: {response.status_code}")

if __name__ == "__main__":
    os.makedirs("data/analysis", exist_ok=True)
    
    # Download USGS 12200500
    download_usgs_data("12200500", "2000-01-01", "2025-12-31", "data/analysis/usgs_12200500_discharge.rdb")
    
    # Download AR Catalog (Skagit Mouth region)
    download_ar_catalog(48.25, -122.5, "data/analysis/ar_catalog_skagit.csv")
