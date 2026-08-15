import daymetpy
import pandas as pd
import numpy as np
from datetime import datetime

def main():
    # Grid of points over Skagit (approx 48.2N to 48.8N, 121W to 122W)
    lats = np.linspace(48.2, 48.8, 5)
    lons = np.linspace(-122.0, -121.0, 5)
    
    years = [2012, 2013, 2020, 2021, 2022, 2023, 2024]
    
    grid_data = []
    for lat in lats:
        for lon in lons:
            print(f"Fetching for {lat}, {lon}...")
            try:
                # daymetpy.daymet_timeseries returns a pandas dataframe
                df = daymetpy.daymet_timeseries(lat, lon, years[0], years[-1])
                grid_data.append(df['prcp'])
            except Exception as e:
                print(f"Failed for {lat}, {lon}: {e}")
    
    if grid_data:
        basin_avg = pd.concat(grid_data, axis=1).mean(axis=1)
        basin_avg.to_csv("data/weather_data/daymet_skagit_precip_daily_basin_backfill.csv")
        print("Done!")

if __name__ == "__main__":
    main()
