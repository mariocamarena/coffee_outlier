import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from data_loader import load_shots
from preprocess import align_and_normalize


def visualize_data_characteristics():
    """Visualize raw data characteristics before preprocessing."""
    shots = load_shots()
    
    # create output directory
    output_dir = Path('raw_data_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # create 2x2 grid for data characteristics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # shot lengths (number of data points)
    ax = axes[0, 0]
    shot_ids = list(shots.keys())
    lengths = [len(shots[sid]) for sid in shot_ids]
    ax.bar(shot_ids, lengths, color='steelblue', alpha=0.7)
    ax.set_xlabel('Shot ID')
    ax.set_ylabel('Number of Data Points')
    ax.set_title('Shot Lengths (Variable)')
    ax.grid(True, alpha=0.3)

    # extraction durations
    ax = axes[0, 1]
    durations = [shots[sid]['espresso_elapsed'].max() for sid in shot_ids]
    ax.bar(shot_ids, durations, color='coral', alpha=0.7)
    ax.set_xlabel('Shot ID')
    ax.set_ylabel('Duration (seconds)')
    ax.set_title('Extraction Durations')
    ax.grid(True, alpha=0.3)

    # scale comparison (box plots)
    ax = axes[1, 0]
    metrics = ['espresso_pressure', 'espresso_flow', 'espresso_temperature_mix']
    data_for_boxplot = []
    for metric in metrics:
        all_values = []
        for sid in shot_ids:
            if metric in shots[sid].columns:
                all_values.extend(shots[sid][metric].values)
        data_for_boxplot.append(all_values)

    ax.boxplot(data_for_boxplot, labels=['Pressure\n(bar)', 'Flow\n(ml/s)', 'Temp Mix\n(°C)'])
    ax.set_ylabel('Raw Values')
    ax.set_title('Scale Differences Across Metrics')
    ax.grid(True, alpha=0.3)

    # time intervals (sampling irregularity)
    ax = axes[1, 1]
    all_intervals = []
    for sid in shot_ids:
        elapsed = shots[sid]['espresso_elapsed'].values
        intervals = np.diff(elapsed)
        all_intervals.extend(intervals)

    ax.hist(all_intervals, bins=30, color='seagreen', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Time Interval (seconds)')
    ax.set_ylabel('Frequency')
    ax.set_title('Sampling Irregularity')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = output_dir / 'data_characteristics.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f'Saved {filename}')


def visualize_shot_comparison():
    """Compare outlier shot (13) vs typical shot (3)."""
    shots = load_shots()
    
    output_dir = Path('raw_data_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # create 2x3 grid: left column = shot 13, right column = shot 10
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    outlier_id = 13
    typical_id = 10

    metrics = [
        ('espresso_pressure', 'Pressure (bar)'),
        ('espresso_flow', 'Flow (ml/s)'),
        ('espresso_temperature_mix', 'Temperature Mix (°C)')
    ]

    for idx, (metric_name, ylabel) in enumerate(metrics):
        # plot left column: shot 13 (outlier)
        ax_left = axes[idx, 0]
        df_outlier = shots[outlier_id]
        ax_left.plot(df_outlier['espresso_elapsed'], df_outlier[metric_name],
                    color='#E74C3C', linewidth=2)
        ax_left.set_ylabel(ylabel)
        ax_left.set_title(f'Shot {outlier_id} (Outlier)')
        ax_left.grid(True, alpha=0.3)

        # plot right column: shot 10 (typical)
        ax_right = axes[idx, 1]
        df_typical = shots[typical_id]
        ax_right.plot(df_typical['espresso_elapsed'], df_typical[metric_name],
                     color='#3498DB', linewidth=2)
        ax_right.set_ylabel(ylabel)
        ax_right.set_title(f'Shot {typical_id} (Typical)')
        ax_right.grid(True, alpha=0.3)

        # add x-label only to bottom row
        if idx == 2:
            ax_left.set_xlabel('Time (s)')
            ax_right.set_xlabel('Time (s)')
    
    plt.tight_layout()
    filename = output_dir / 'shot_comparison.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f'Saved {filename}')


def visualize_shot_comparison_preprocessed():
    """Compare outlier shot (13) vs typical shot (10) after preprocessing."""
    shots = load_shots()
    aligned, stats = align_and_normalize(shots)
    
    output_dir = Path('raw_data_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # create 2x3 grid: left column = shot 13, right column = shot 10
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    outlier_id = 13
    typical_id = 10

    metrics = [
        ('espresso_pressure', 'Pressure (z-score)'),
        ('espresso_flow', 'Flow (z-score)'),
        ('espresso_temperature_mix', 'Temperature Mix (z-score)')
    ]

    for idx, (metric_name, ylabel) in enumerate(metrics):
        # plot left column: shot 13 (outlier)
        ax_left = axes[idx, 0]
        df_outlier = aligned[outlier_id]
        time_points = np.arange(len(df_outlier))
        ax_left.plot(time_points, df_outlier[metric_name],
                    color='#E74C3C', linewidth=2)
        ax_left.set_ylabel(ylabel)
        ax_left.set_title(f'Shot {outlier_id} (Outlier) - Preprocessed')
        ax_left.grid(True, alpha=0.3)
        ax_left.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)

        # plot right column: shot 10 (typical)
        ax_right = axes[idx, 1]
        df_typical = aligned[typical_id]
        ax_right.plot(time_points, df_typical[metric_name],
                     color='#3498DB', linewidth=2)
        ax_right.set_ylabel(ylabel)
        ax_right.set_title(f'Shot {typical_id} (Typical) - Preprocessed')
        ax_right.grid(True, alpha=0.3)
        ax_right.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)

        # add x-label only to bottom row
        if idx == 2:
            ax_left.set_xlabel('Time Point (0-200)')
            ax_right.set_xlabel('Time Point (0-200)')
    
    plt.tight_layout()
    filename = output_dir / 'shot_comparison_preprocessed.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f'Saved {filename}')


def visualize_all_shots_overlay():
    """Overlay all shots for each metric, highlighting shot 13."""
    shots = load_shots()
    
    output_dir = Path('raw_data_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # create 2x3 grid for 6 key metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    metrics = [
        ('espresso_pressure', 'Pressure (bar)'),
        ('espresso_flow', 'Flow (ml/s)'),
        ('espresso_temperature_basket', 'Temperature Basket (°C)'),
        ('espresso_temperature_mix', 'Temperature Mix (°C)'),
        ('espresso_water_dispensed', 'Water Dispensed (ml)'),
        ('espresso_resistance', 'Resistance'),
    ]

    outlier_id = 13

    for idx, (metric_name, ylabel) in enumerate(metrics):
        ax = axes[idx]

        # check if metric exists
        if metric_name not in shots[1].columns:
            continue

        # plot all shots in gray
        for shot_id, df in shots.items():
            if shot_id == outlier_id:
                continue
            ax.plot(df['espresso_elapsed'], df[metric_name],
                   color='#CCCCCC', alpha=0.5, linewidth=1)

        # plot shot 13 in red on top
        df_outlier = shots[outlier_id]
        ax.plot(df_outlier['espresso_elapsed'], df_outlier[metric_name],
               color='#E74C3C', linewidth=2.5, label=f'Shot {outlier_id}')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.tight_layout()
    filename = output_dir / 'all_shots_overlay.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f'Saved {filename}')


def visualize_all():
    """Generate all raw data visualizations."""
    print('Generating raw data visualizations...')
    print()

    print('Creating data characteristics plot...')
    visualize_data_characteristics()

    print('Creating shot comparison plot (raw)...')
    visualize_shot_comparison()

    print('Creating shot comparison plot (preprocessed)...')
    visualize_shot_comparison_preprocessed()

    print('Creating all shots overlay plot...')
    visualize_all_shots_overlay()

    print()
    print('All visualizations saved to raw_data_visualizations/')


if __name__ == "__main__":
    visualize_all()

