import os
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = "/data0/hernanqd/plots_code/skagit_basin_de"
DATA_CSV = os.path.join(BASE_DIR, "cumulative_precipitation_plot/event_precipitation_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "cumulative_precipitation_plot")


def plot_non_ar_events():
    """Plot cumulative precipitation for top 6 non-AR events"""
    print("Loading precipitation data...")
    data_df = pd.read_csv(DATA_CSV)

    # Filter for non-AR events only (ar_scale == 0)
    non_ar_data = data_df[data_df['ar_scale'] == 0].copy()

    # Get unique non-AR event dates
    unique_events = non_ar_data['event_date'].unique()

    # Create plots for top 6 events
    top_events = unique_events[:6]

    print(f"Creating plot for top {len(top_events)} non-AR events...")

    # Calculate max cumulative precipitation across all non-AR events for consistent y-axis
    products = ['prism', 'pnnl', 'daymet', 'conus', 'ucla', 'gridmet']
    max_cumsum = 0
    for event_date in unique_events:
        event_data = non_ar_data[non_ar_data['event_date'] == event_date].copy()
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
        event_data = non_ar_data[non_ar_data['event_date'] == event_date].copy()
        event_data['window_date'] = pd.to_datetime(event_data['window_date'])
        event_data = event_data.sort_values('window_date')
        event_data.set_index('window_date', inplace=True)

        # Plot each product
        for prod_idx, product in enumerate(products):
            if product in event_data.columns:
                cumsum = event_data[product].cumsum()
                ax.plot(cumsum.index, cumsum, label=product.upper(),
                       marker=markers[prod_idx], linewidth=2, markersize=5,
                       color=colors[prod_idx], alpha=0.8)

        ax.set_title(f'Event: {event_date}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Cumulative Precipitation (mm)', fontsize=10)
        ax.set_ylim(0, ylim_max)
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Cumulative Precipitation for Top 6 Non-AR Events', fontsize=16, fontweight='bold')

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'non_ar_events_cumulative_precipitation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_non_ar_events()
