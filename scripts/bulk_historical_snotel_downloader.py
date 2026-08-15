import subprocess
import calendar
import time
import os

START_YEAR = 1990
END_YEAR = 2013
OUTPUT_DIR = "/data0/skagit_met/data_transfer/data/snotel"
BASE_DIR = "/home/nksp2/skagit/skagit_2/skagit-met"

def download_month(year, month, retries=5):
    start_date = f"{year}-{month:02d}-01"
    last_day = calendar.monthrange(year, month)[1]
    end_date = f"{year}-{month:02d}-{last_day:02d}"
    
    print(f"--- Processing SNOTEL for {start_date} to {end_date} ---", flush=True)
    cmd = [
        "pixi", "run", "-e", "data-download", "python", "scripts/snotel_downloader.py",
        "--startDate", start_date,
        "--endDate", end_date,
        "--outputDir", OUTPUT_DIR,
        "--frequency", "daily",
        "--variables", "PRECIPITATION,ACCUMULATED PRECIPITATION"
    ]
    
    for attempt in range(retries):
        try:
            # We want to see the output from snotel_downloader.py
            result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
            if result.returncode == 0:
                if "No variable data found" in result.stdout:
                    print(f"  Info: No data found for {start_date}", flush=True)
                else:
                    print(f"  Success: {start_date}", flush=True)
                    print(result.stdout, flush=True)
                return True
            else:
                wait = (2 ** attempt) * 5
                print(f"  Attempt {attempt+1} failed ({result.returncode}) for {start_date}. Error: {result.stderr.strip()}. Retrying in {wait}s...", flush=True)
                time.sleep(wait)
        except Exception as e:
            print(f"  Exception on attempt {attempt+1}: {e}", flush=True)
            time.sleep(5)
    
    return False

if __name__ == "__main__":
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            last_day = calendar.monthrange(year, month)[1]
            # Check if file already exists
            expected_file = os.path.join(OUTPUT_DIR, f"{year}-{month:02d}-01_{year}-{month:02d}-{last_day:02d}_SNOTEL_daily_data.zarr")
            if os.path.exists(expected_file):
                # print(f"Skipping {year}-{month:02d} (already exists)")
                continue
            
            download_month(year, month)
