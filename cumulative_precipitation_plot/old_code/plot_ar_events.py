import os
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = "/data0/hernanqd/plots_code/skagit_basin_de"
DATA_CSV = os.path.join(BASE_DIR, "cumulative_precipitation_plot/event_precipitation_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "cumulative_precipitation_plot")


def plot_ar_events():
    """Plot cumulative precipitation for top 20 AR events"""
    print("Loading precipitation data...")
    data_df = pd.read_csv(DATA_CSV)

    # Filter for AR events only (ar_scale != 0)
    ar_data = data_df[data_df['ar_scale'] != 0].copy()

    # Get unique AR event dates
    unique_events = ar_data['event_date'].unique()

    # Create plots for top 7 events, excluding 2025-12-11
    top_events = unique_events[:7]
    top_events = [e for e in top_events if e != '2025-12-11'] #we don't have enough precip data for this event

    print(f"Creating plot for top {len(top_events)} AR events...")

    # Calculate max cumulative precipitation across all AR events for consistent y-axis
    products = ['prism', 'pnnl', 'daymet', 'conus', 'ucla', 'gridmet']
    max_cumsum = 0
    for event_date in unique_events:
        event_data = ar_data[ar_data['event_date'] == event_date].copy()
        event_data['window_date'] = pd.to_datetime(event_data['window_date'])
        event_data = event_data.sort_values('window_date')
        for product in products:
            if product in event_data.columns:
                cumsum = event_data[product].cumsum()
                max_cumsum = max(max_cumsum, cumsum.max())
    ylim_max = max_cumsum * 1.05

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    for idx, event_date in enumerate(top_events):
        ax = axes[idx]

        # Get data for this event
        event_data = ar_data[ar_data['event_date'] == event_date].copy()
        event_data['window_date'] = pd.to_datetime(event_data['window_date'])
        event_data = event_data.sort_values('window_date')
        event_data.set_index('window_date', inplace=True)

        # Get AR scale
        ar_scale = event_data['ar_scale'].iloc[0]

        # Plot each product
        for prod_idx, product in enumerate(products):
            if product in event_data.columns:
                cumsum = event_data[product].cumsum()
                ax.plot(cumsum.index, cumsum, label=product.upper(),
                       marker=markers[prod_idx], linewidth=2, markersize=5,
                       color=colors[prod_idx], alpha=0.8)

        ar_label = f' (AR Scale: {ar_scale})'
        ax.set_title(f'Event: {event_date}{ar_label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Cumulative Precipitation (mm)', fontsize=10)
        ax.set_ylim(0, ylim_max)
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Cumulative Precipitation for Top AR Events', fontsize=16, fontweight='bold')

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'ar_events_cumulative_precipitation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_ar_events()
