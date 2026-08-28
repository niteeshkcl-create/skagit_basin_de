import os
import pandas as pd
import xarray as xr

VAULT_DIR = "/data0/skagit_met/data_transfer/data"
PRISM_DIR = os.path.join(VAULT_DIR, "PRISM")

# Scan all decade directories for zarr files
decade_folders = [d for d in os.listdir(PRISM_DIR) if os.path.isdir(os.path.join(PRISM_DIR, d))]
decade_folders.sort()

print("Scanning PRISM zarr files for data coverage...")
print("="*80)

coverage_by_year = {}
all_years = []

for decade in decade_folders:
    decade_path = os.path.join(PRISM_DIR, decade)
    zarr_files = [f for f in os.listdir(decade_path) if f.endswith('.zarr') and 'daily_4km' in f]

    print(f"\nDecade: {decade}")

    for zarr_file in sorted(zarr_files):
        zarr_path = os.path.join(decade_path, zarr_file)

        try:
            ds = xr.open_zarr(zarr_path, consolidated=False)
            dates = pd.to_datetime(ds.time.values)

            year = int(zarr_file.split('-')[0])
            start_date = dates.min()
            end_date = dates.max()
            expected_dates = pd.date_range(start=pd.Timestamp(year, 1, 1), end=pd.Timestamp(year, 12, 31), freq='D')

            available_count = len(dates)
            expected_count = len(expected_dates)
            coverage_pct = (available_count / expected_count * 100) if expected_count > 0 else 0

            # Find missing dates
            missing_dates = [d for d in expected_dates if d not in dates.values]

            coverage_by_year[year] = {
                'decade': decade,
                'available': available_count,
                'expected': expected_count,
                'coverage_pct': coverage_pct,
                'start_date': start_date,
                'end_date': end_date,
                'missing_dates': missing_dates,
                'first_date': start_date,
                'last_date': end_date
            }

            all_years.append(year)

            if coverage_pct == 100:
                status = "✓ COMPLETE"
            else:
                status = f"✗ {available_count}/{expected_count} ({coverage_pct:.1f}%)"

            print(f"  {year}: {status} [{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}]")

            ds.close()

        except Exception as e:
            print(f"  Error reading {zarr_file}: {e}")

all_years.sort()

# Print detailed summary
print("\n" + "="*80)
print("COVERAGE SUMMARY")
print("="*80)

complete = sum(1 for d in coverage_by_year.values() if d['coverage_pct'] == 100)
partial = sum(1 for d in coverage_by_year.values() if 0 < d['coverage_pct'] < 100)
missing = sum(1 for d in coverage_by_year.values() if d['coverage_pct'] == 0)

print(f"Complete coverage (100%): {complete} years")
print(f"Partial coverage (1-99%): {partial} years")
print(f"No coverage (0%): {missing} years")
print(f"\nYears available: {all_years[0]} to {all_years[-1]} ({len(all_years)} years total)")

# Show years with incomplete coverage
incomplete = [(y, coverage_by_year[y]) for y in all_years if coverage_by_year[y]['coverage_pct'] < 100]
if incomplete:
    print(f"\n" + "="*80)
    print("YEARS WITH INCOMPLETE COVERAGE")
    print("="*80)
    for year, data in incomplete:
        print(f"\n{year}: {data['coverage_pct']:.1f}% ({data['available']}/{data['expected']} dates)")
        print(f"  Range: {data['first_date'].strftime('%Y-%m-%d')} to {data['last_date'].strftime('%Y-%m-%d')}")
        if len(data['missing_dates']) <= 20:
            print(f"  Missing dates ({len(data['missing_dates'])}): {', '.join([d.strftime('%Y-%m-%d') for d in data['missing_dates']])}")
        else:
            missing_by_month = {}
            for d in data['missing_dates']:
                key = d.strftime('%Y-%m')
                if key not in missing_by_month:
                    missing_by_month[key] = []
                missing_by_month[key].append(d)

            print(f"  Missing dates ({len(data['missing_dates'])}): Gaps by month:")
            for month_key in sorted(missing_by_month.keys()):
                dates_in_month = missing_by_month[month_key]
                print(f"    {month_key}: {len(dates_in_month)} missing days")
