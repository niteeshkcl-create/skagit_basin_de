import xarray as xr
import pandas as pd
import os

prism_path = "/data0/skagit_met/data_transfer/data/PRISM/2001-2010/2003-01-01_2003-12-31_daily_4km_PRISM_data.zarr"

try:
    ds = xr.open_zarr(prism_path, consolidated=False)
    print("PRISM 2003 Dataset opened successfully")
    print(f"Time dimension size: {len(ds.time)}")

    # Convert time to datetime and check the range
    time_vals = pd.to_datetime(ds.time.values)
    print(f"Time range: {time_vals.min()} to {time_vals.max()}")

    # Check for the event window
    event_start = pd.Timestamp('2003-10-14')
    event_end = pd.Timestamp('2003-11-03')

    event_dates = pd.date_range(event_start, event_end, freq='D')
    print(f"\nEvent window: {event_start} to {event_end}")
    print(f"Expected {len(event_dates)} days")

    # Find which dates are in the dataset
    available_in_window = [d for d in event_dates if d in time_vals]
    missing_in_window = [d for d in event_dates if d not in time_vals]

    print(f"\nAvailable dates in window: {len(available_in_window)}")
    print(f"Missing dates in window: {len(missing_in_window)}")

    if missing_in_window:
        print("\nMissing dates:")
        for d in missing_in_window:
            print(f"  {d.strftime('%Y-%m-%d')}")

    ds.close()
except Exception as e:
    print(f"Error: {e}")
