import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# CANONICAL SETTINGS
# ============================================================
DATASET_ORDER = [
    "PRISM 4km",
    "Daymet-PC",
    "gridMET",
    "HRRR",
    "UCLA ERA5 d02",
    "PNNL hist",
    "CONUS404",
]

COLORS = {
    "PRISM 4km":     "#1B9E77",
    "Daymet-PC":     "#7570B3",
    "gridMET":       "#E7298A",
    "HRRR":          "#1F78B4",
    "UCLA ERA5 d02": "#E6AB02",
    "PNNL hist":     "#D95F02",
    "CONUS404":      "#666666",
}

# Mapping for PPT CSV which uses older names
PPT_NAME_MAP = {
    "PRISM": "PRISM 4km",
    "Daymet v4 (Planetary Computer)": "Daymet-PC",
    "HRRR (f06)": "HRRR",
    "HRRR (f01)": "HRRR",
    "UCLA ERA5 WRF d02 (daily NetCDF)": "UCLA ERA5 d02",
    "PNNL (historical)": "PNNL hist",
    "CONUS404 (daily-osn)": "CONUS404",
}

TEMP_CSV = "/data0/skagit_met/data_transfer/data/derived/wy_mean_temp_skagit_regions_1980_2024_ALL.csv"
PPT_CSV = "/data0/skagit_met/data_transfer/data/derived/wy_total_precip_skagit_regions_1983_2024_ALL.csv"

OUT_DIR = Path("/home/nksp2/skagit/skagit_2/skagit-met/analysis/event_analysis/atmospheric_rivers/graphs_2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_clean_data():
    # 1. Load Temperature
    df_temp = pd.read_csv(TEMP_CSV)
    df_temp = df_temp[df_temp["region"] == "Full Skagit"].copy()
    
    # 2. Load Precipitation
    df_ppt = pd.read_csv(PPT_CSV)
    df_ppt = df_ppt[df_ppt["region"] == "Full Skagit"].copy()
    df_ppt["dataset"] = df_ppt["dataset"].map(lambda x: PPT_NAME_MAP.get(x, x))

    # ----------------------------
    # CLEANING
    # ----------------------------
    
    # A) Remove zeros after 2020
    df_temp = df_temp[~((df_temp["year"] > 2020) & (df_temp["temp_c"] == 0))]
    df_ppt = df_ppt[~((df_ppt["year"] > 2020) & (df_ppt["mm"] == 0))]

    # B) Remove specific outliers previously identified
    df_ppt = df_ppt[~((df_ppt["dataset"] == "HRRR") & (df_ppt["year"] == 2015))]
    df_temp = df_temp[~((df_temp["dataset"] == "HRRR") & (df_temp["year"] == 2014))]
    df_temp = df_temp[~((df_temp["dataset"] == "Daymet-PC") & (df_temp["year"] == 2021))]

    # C) Remove points below thresholds (New User Request)
    # "remove the point which goes below 2 in temp and below 1250 in precipitataion"
    df_temp = df_temp[df_temp["temp_c"] >= 2]
    df_ppt = df_ppt[df_ppt["mm"] >= 1250]

    return df_temp, df_ppt

def plot_combined_panel(df_temp, df_ppt, out_filename, start_year=None):
    if start_year:
        df_temp = df_temp[df_temp["year"] >= start_year].copy()
        df_ppt = df_ppt[df_ppt["year"] >= start_year].copy()

    all_years = sorted(list(set(df_temp["year"]) | set(df_ppt["year"])))
    if not all_years: return
    
    xticks = np.arange(min(all_years), max(all_years) + 1, 2)
    xlims = (min(all_years)-1, max(all_years)+1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
    
    for ds in DATASET_ORDER:
        d = df_temp[df_temp["dataset"] == ds].sort_values("year")
        if not d.empty:
            ax1.plot(d["year"], d["temp_c"], color=COLORS[ds], linewidth=2.5, label=ds)
        
        d2 = df_ppt[df_ppt["dataset"] == ds].sort_values("year")
        if not d2.empty:
            ax2.plot(d2["year"], d2["mm"], color=COLORS[ds], linewidth=2.5, label=ds)
    
    ax1.set_title("Full Skagit | Water-Year Mean Temperature", fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel("Temperature (°C)", fontsize=13)
    ax1.grid(True, which='both', color='grey', linestyle='-', alpha=0.4, linewidth=1.5)
    ax1.set_ylim(0, 9)
    
    ax2.set_title("Full Skagit | Water-Year Total Precipitation", fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel("Precipitation (mm)", fontsize=13)
    ax2.set_xlabel("Water Year", fontsize=13)
    ax2.grid(True, which='both', color='grey', linestyle='-', alpha=0.4, linewidth=1.5)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks, rotation=45)
    ax2.set_xlim(xlims)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    all_h = []
    all_l = []
    for h, l in zip(h1 + h2, l1 + l2):
        if l not in all_l:
            all_h.append(h)
            all_l.append(l)
    
    # Sort legend by DATASET_ORDER
    sorted_pairs = []
    for ds in DATASET_ORDER:
        if ds in all_l:
            idx = all_l.index(ds)
            sorted_pairs.append((all_h[idx], all_l[idx]))
    
    final_h = [p[0] for p in sorted_pairs]
    final_l = [p[1] for p in sorted_pairs]
    
    fig.legend(final_h, final_l, loc='lower center', ncol=len(final_l), bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=12)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_individual(df_temp, df_ppt):
    all_years = sorted(list(set(df_temp["year"]) | set(df_ppt["year"])))
    if not all_years: return
    
    xticks = np.arange(min(all_years), max(all_years) + 1, 2)
    xlims = (min(all_years)-1, max(all_years)+1)

    # ----------------------------
    # INDIVIDUAL TEMP PLOT
    # ----------------------------
    fig, ax = plt.subplots(figsize=(18, 8))
    for ds in DATASET_ORDER:
        d = df_temp[df_temp["dataset"] == ds].sort_values("year")
        if not d.empty:
            ax.plot(d["year"], d["temp_c"], color=COLORS[ds], linewidth=2.5, label=ds)
    
    ax.set_title("Full Skagit | Water-Year Mean Temperature", fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel("Temperature (°C)", fontsize=14)
    ax.set_xlabel("Water Year", fontsize=14)
    ax.grid(True, which='both', color='grey', linestyle='-', alpha=0.4, linewidth=1.5)
    ax.set_ylim(0, 9)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=45)
    ax.set_xlim(xlims)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "wy_mean_temp_Full_Skagit_standardized.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ----------------------------
    # INDIVIDUAL PRECIP PLOT
    # ----------------------------
    fig, ax = plt.subplots(figsize=(18, 8))
    for ds in DATASET_ORDER:
        d = df_ppt[df_ppt["dataset"] == ds].sort_values("year")
        if not d.empty:
            ax.plot(d["year"], d["mm"], color=COLORS[ds], linewidth=2.5, label=ds)
    
    ax.set_title("Full Skagit | Water-Year Total Precipitation", fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel("Precipitation (mm)", fontsize=14)
    ax.set_xlabel("Water Year", fontsize=14)
    ax.grid(True, which='both', color='grey', linestyle='-', alpha=0.4, linewidth=1.5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=45)
    ax.set_xlim(xlims)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "wy_total_precip_Full_Skagit_standardized.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_all():
    df_temp, df_ppt = load_and_clean_data()
    
    # 1. Combined (Full timeline available)
    plot_combined_panel(df_temp, df_ppt, OUT_DIR / "wy_temp_precip_combined_standardized.png")
    
    # 2. Combined (Starting from 1983)
    plot_combined_panel(df_temp, df_ppt, OUT_DIR / "wy_temp_precip_combined_standardized_from1983.png", start_year=1983)
    
    # 3. Individual plots
    plot_individual(df_temp, df_ppt)

    print(f"Generated all standardized plots in {OUT_DIR}")

if __name__ == "__main__":
    plot_all()
