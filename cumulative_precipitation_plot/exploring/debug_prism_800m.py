import xarray as xr
import pandas as pd
import os

# Check 800m product
prism_path_800 = "/data0/skagit_met/data_transfer/data/prism_ppt_800m/2003-01-01_2003-12-31_daily_800m_PRISM_data.zarr"

print("Checking 800m PRISM product...")
if os.path.exists(prism_path_800):
    try:
        ds = xr.open_zarr(prism_path_800, consolidated=False)
        print("✓ 800m PRISM 2003 Dataset exists")
        print(f"  Time dimension size: {len(ds.time)}")

        # Convert time to datetime and check the range
        time_vals = pd.to_datetime(ds.time.values)
        print(f"  Time range: {time_vals.min()} to {time_vals.max()}")

        # Check for the event window
        event_start = pd.Timestamp('2003-10-14')
        event_end = pd.Timestamp('2003-11-03')

        event_dates = pd.date_range(event_start, event_end, freq='D')

        # Find which dates are in the dataset
        available_in_window = [d for d in event_dates if d in time_vals]
        missing_in_window = [d for d in event_dates if d not in time_vals]

        print(f"\n  Event window availability:")
        print(f"    Available: {len(available_in_window)}/21 days")
        print(f"    Missing: {len(missing_in_window)} days")

        if missing_in_window:
            print(f"    Missing dates: {[d.strftime('%m-%d') for d in missing_in_window]}")

        ds.close()
    except Exception as e:
        print(f"✗ Error reading 800m product: {e}")
else:
    print(f"✗ 800m product not found at {prism_path_800}")

print("\n" + "="*60)
print("For comparison - 4km product had:")
print("  Available: 16/21 days")
print("  Missing: 2003-10-17 to 2003-10-21")
