import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

BASE_DIR = "/home/nksp2/skagit/skagit_2/skagit-met"
DATA_DIR = os.path.join(BASE_DIR, "experiments_4")
RESULTS_CSV = os.path.join(DATA_DIR, "multi_product_bulk_bias_data.csv")
OUT_DIR = os.path.join(BASE_DIR, "refined_plots")
OUT_IMG = os.path.join(OUT_DIR, "refined_grouped_bias_3_metrics_2_periods.png")

def plot_refined_bias():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(RESULTS_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['ar_scale'])
    df = df[(df['ar_scale'] >= 0) & (df['ar_scale'] <= 5)]
    
    # Combined Buckets: combining first 3 streamflow categories into '< 30k'
    bins_sf = [0, 30000, 40000, 50000, 60000, 70000, 80000, float('inf')]
    labels_sf = ['< 30k', '30k-40k', '40k-50k', '50k-60k', '60k-70k', '70k-80k', '> 80k']
    df['Streamflow Bucket (cfs)'] = pd.cut(df['discharge_cfs'], bins=bins_sf, labels=labels_sf)
    
    bins_pr = [-1, 30, 60, 90, 120, 150, float('inf')]
    labels_pr = ['< 30mm', '30-60mm', '60-90mm', '90-120mm', '120-150mm', '> 150mm']
    df['Precipitation Bucket (mm)'] = pd.cut(df['prism_3d_tot'], bins=bins_pr, labels=labels_pr)
    
    # Calculate Bias (Product - PRISM Baseline)
    df['PNNL'] = df['pnnl_3d_tot'] - df['prism_3d_tot']
    df['CONUS404'] = df['conus_3d_tot'] - df['prism_3d_tot']
    df['UCLA'] = df['ucla_3d_tot'] - df['prism_3d_tot']
    df['GridMET'] = df['gridmet_3d_tot'] - df['prism_3d_tot']
    df['HRRR'] = df['hrrr_3d_tot'] - df['prism_3d_tot']
    
    melted_df = df.melt(id_vars=['date', 'ar_scale', 'Streamflow Bucket (cfs)', 'Precipitation Bucket (mm)', 'prism_3d_tot'], 
                        value_vars=['PNNL', 'CONUS404', 'UCLA', 'GridMET', 'HRRR'],
                        var_name='Product', 
                        value_name='Bias')
    melted_df = melted_df.dropna(subset=['Bias'])
    
    products = ['PNNL', 'CONUS404', 'UCLA', 'GridMET', 'HRRR']
    palette = {
        'PNNL': '#d95f02', 
        'CONUS404': '#e7298a', 
        'UCLA': '#66a61e', 
        'GridMET': '#e6ab02', 
        'HRRR': '#a6761d'
    }
    
    # Historical Period (1981-2020) without HRRR
    df_p1 = melted_df[(melted_df['date'].dt.year >= 1981) & (melted_df['date'].dt.year <= 2020)]
    df_p1 = df_p1[df_p1['Product'] != 'HRRR']
    products_p1 = [p for p in products if p != 'HRRR']
    
    # Modern Period (2014-Present) with HRRR
    df_p2 = melted_df[melted_df['date'].dt.year >= 2014]
    products_p2 = products

    def get_ar_info(data, prod_list):
        stats = data.groupby('ar_scale')['prism_3d_tot'].agg(['count', 'min', 'max']).reset_index()
        num_prods = len(prod_list)
        labels = [f"AR {int(row['ar_scale'])}.0\nN={int(row['count']/num_prods)}\n{int(row['min'])}-{int(row['max'])}mm" for i, row in stats.iterrows()]
        return labels

    def get_bucket_info(data, col, prod_list):
        stats = data.groupby(col, observed=False)['Bias'].agg(['count']).reset_index()
        num_prods = len(prod_list)
        labels = [f"{row[col]}\nN={int(row['count']/num_prods)}" for i, row in stats.iterrows()]
        return labels

    # Creating subplots sharing y-axis per row
    fig, axes = plt.subplots(3, 2, figsize=(28, 26), facecolor='#ffffff', sharey='row')
    sns.set_theme(style="whitegrid")
    
    for col_idx, (data, title_suffix, h_order) in enumerate([(df_p1, "(1981-2020)", products_p1), (df_p2, "(2014-Present)", products_p2)]):
        # Row 1: AR Scale
        labels_ar = get_ar_info(data, h_order)
        sns.boxplot(data=data, x='ar_scale', y='Bias', hue='Product', palette=palette, hue_order=h_order, linewidth=1.2, fliersize=2, ax=axes[0, col_idx])
        axes[0, col_idx].set_xticks(range(len(labels_ar)))
        axes[0, col_idx].set_xticklabels(labels_ar, fontsize=12)
        axes[0, col_idx].set_title(f"Bias by AR Scale {title_suffix}", fontsize=18, weight='bold')
        axes[0, col_idx].set_xlabel("Atmospheric River Scale (0-5)", fontsize=14)
        
        # Row 2: Streamflow Bucket
        labels_sf_info = get_bucket_info(data, 'Streamflow Bucket (cfs)', h_order)
        sns.boxplot(data=data, x='Streamflow Bucket (cfs)', y='Bias', hue='Product', palette=palette, hue_order=h_order, linewidth=1.2, fliersize=2, ax=axes[1, col_idx])
        axes[1, col_idx].set_xticks(range(len(labels_sf_info)))
        axes[1, col_idx].set_xticklabels(labels_sf_info, fontsize=12)
        axes[1, col_idx].set_title(f"Bias by Streamflow Intensity {title_suffix}", fontsize=18, weight='bold')
        axes[1, col_idx].set_xlabel("Streamflow Bucket (cfs)", fontsize=14)
        
        # Row 3: Precipitation Bucket
        labels_pr_info = get_bucket_info(data, 'Precipitation Bucket (mm)', h_order)
        sns.boxplot(data=data, x='Precipitation Bucket (mm)', y='Bias', hue='Product', palette=palette, hue_order=h_order, linewidth=1.2, fliersize=2, ax=axes[2, col_idx])
        axes[2, col_idx].set_xticks(range(len(labels_pr_info)))
        axes[2, col_idx].set_xticklabels(labels_pr_info, fontsize=12)
        axes[2, col_idx].set_title(f"Bias by PRISM Precip Intensity {title_suffix}", fontsize=18, weight='bold')
        axes[2, col_idx].set_xlabel("Precipitation Bucket (mm)", fontsize=14)

    for r_idx in range(3):
        for c_idx in range(2):
            ax = axes[r_idx, c_idx]
            ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.set_ylabel("Bias (mm)", fontsize=14)
            # Display legend on each plot but keep layout clean
            ax.legend(title="Product", loc='upper left', bbox_to_anchor=(1, 1), fontsize=12, title_fontsize=13)

    plt.suptitle("Enhanced Multi-Product Precipitation Bias Analysis\n(Product - PRISM Baseline)", fontsize=24, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=300, bbox_inches='tight')
    print(f"Saved refined bias plot to {OUT_IMG}")

if __name__ == "__main__":
    plot_refined_bias()
