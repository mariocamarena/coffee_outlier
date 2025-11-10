import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import load_shots
from outlier_detection import compute_outlier_scores, find_most_anomalous_shot
from preprocess import align_and_normalize
from features import compute_features


def plot_all_profiles():
    """Plot brewing profiles for all shots (individual files)."""
    shots = load_shots()
    
    # key metrics to plot
    metrics = [
        ('espresso_pressure', 'Pressure (bar)'),
        ('espresso_flow', 'Flow (ml/s)'),
        ('espresso_temperature_basket', 'Temperature Basket (°C)'),
        ('espresso_temperature_mix', 'Temperature Mix (°C)'),
        ('espresso_water_dispensed', 'Water Dispensed (ml)'),
        ('espresso_resistance', 'Resistance'),
    ]
    
    for metric_name, ylabel in metrics:
        # check if metric exists
        if metric_name not in shots[1].columns:
            continue
        
        plt.figure(figsize=(10, 6))
        
        # plot all shots
        for shot_id, df in shots.items():
            plt.plot(df['espresso_elapsed'], df[metric_name], 
                    alpha=0.7, label=f'Shot {shot_id}')
        
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} Profiles')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # save figure
        filename = f'results/{metric_name}_profiles.png'
        plt.savefig(filename)
        plt.close()
        print(f'Saved {filename}')
    
    print('All individual plots saved to results/')


def plot_all_profiles_combined():
    """Plot all brewing profiles in a single combined image."""
    shots = load_shots()
    
    # key metrics to plot
    metrics = [
        ('espresso_pressure', 'Pressure (bar)'),
        ('espresso_flow', 'Flow (ml/s)'),
        ('espresso_temperature_basket', 'Temperature Basket (°C)'),
        ('espresso_temperature_mix', 'Temperature Mix (°C)'),
        ('espresso_water_dispensed', 'Water Dispensed (ml)'),
        ('espresso_resistance', 'Resistance'),
    ]
    
    # create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (metric_name, ylabel) in enumerate(metrics):
        ax = axes[idx]
        
        # check if metric exists
        if metric_name not in shots[1].columns:
            ax.text(0.5, 0.5, f'{ylabel}\n(Not Available)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(ylabel)
            continue
        
        # plot all shots
        for shot_id, df in shots.items():
            ax.plot(df['espresso_elapsed'], df[metric_name], 
                   alpha=0.6, linewidth=1.5, label=f'Shot {shot_id}')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8, ncol=2)
    
    plt.suptitle('All Espresso Shot Profiles', fontsize=16, y=0.995)
    plt.tight_layout()
    
    # save figure
    filename = 'results/all_profiles_combined.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f'Saved {filename}')


def plot_outlier_verification_individual():
    """Generate individual outlier verification plots and save statistics to text file."""
    # load data and compute outlier
    shots = load_shots()
    aligned, stats = align_and_normalize(shots)
    features = compute_features(aligned)
    scores = compute_outlier_scores(features)
    outlier_id = find_most_anomalous_shot(scores)
    
    # key metrics for verification - ALL 6 metrics
    metrics = [
        ('espresso_pressure', 'Pressure (bar)'),
        ('espresso_flow', 'Flow (ml/s)'),
        ('espresso_temperature_basket', 'Temperature Basket (°C)'),
        ('espresso_temperature_mix', 'Temperature Mix (°C)'),
        ('espresso_water_dispensed', 'Water Dispensed (ml)'),
        ('espresso_resistance', 'Resistance'),
    ]
    
    # Generate individual plots for each metric
    for metric_name, ylabel in metrics:
        # check if metric exists
        if metric_name not in shots[1].columns:
            continue
        
        plt.figure(figsize=(10, 6))
        
        # plot all non-outlier shots in blue/light blue
        for shot_id, df in shots.items():
            if shot_id == outlier_id:
                continue
            plt.plot(df['espresso_elapsed'], df[metric_name], 
                    color='#6495ED', alpha=0.4, linewidth=1.2)
        
        # plot outlier in bold red on top
        df_outlier = shots[outlier_id]
        plt.plot(df_outlier['espresso_elapsed'], df_outlier[metric_name], 
                color='#DC143C', linewidth=2.5, label=f'Shot {outlier_id} (OUTLIER)',
                zorder=10)
        
        plt.xlabel('Time (s)', fontsize=11)
        plt.ylabel(ylabel, fontsize=11)
        plt.title(f'{ylabel} - Outlier Highlighted', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()
        
        # save individual metric plot
        filename = f'results/outlier_verification_{metric_name}.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f'Saved {filename}')
    
    # Generate outlier scores bar chart
    plt.figure(figsize=(12, 6))
    sorted_scores = scores.sort_values(ascending=False)
    colors = ['#DC143C' if shot_id == outlier_id else '#6495ED' 
              for shot_id in sorted_scores.index]
    
    bars = plt.bar(range(len(sorted_scores)), sorted_scores.values, color=colors)
    plt.xlabel('Shot ID', fontsize=12, fontweight='bold')
    plt.ylabel('Outlier Score', fontsize=12, fontweight='bold')
    plt.title('Outlier Scores (L2 Norm of Z-scores)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(sorted_scores)), 
               [f'{shot_id}' for shot_id in sorted_scores.index], 
               fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for i, (shot_id, score) in enumerate(sorted_scores.items()):
        plt.text(i, score, f'{score:.2f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold' if shot_id == outlier_id else 'normal')
    
    plt.tight_layout()
    filename = 'results/outlier_scores_chart.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f'Saved {filename}')
    
    # Compute statistics
    outlier_score = scores[outlier_id]
    second_highest = sorted_scores.iloc[1]
    second_highest_id = sorted_scores.index[1]
    mean_score = scores.mean()
    median_score = scores.median()
    std_score = scores.std()
    
    # Calculate percentage differences
    pct_above_second = ((outlier_score - second_highest) / second_highest) * 100
    pct_above_median = ((outlier_score - median_score) / median_score) * 100
    std_deviations = (outlier_score - mean_score) / std_score
    
    # Create statistics text and save to file
    stats_text = f"""OUTLIER DETECTION STATISTICS
{'='*70}

Shot {outlier_id} Outlier Score: {outlier_score:.4f}

COMPARISON TO OTHER SHOTS:
{'─'*70}
  • {pct_above_second:.1f}% higher than 2nd highest (Shot {second_highest_id}: {second_highest:.4f})
  • {pct_above_median:.1f}% higher than median ({median_score:.4f})
  • {std_deviations:.2f} standard deviations above mean ({mean_score:.4f})

SCORE DISTRIBUTION:
{'─'*70}
  • Mean Score:   {mean_score:.4f}
  • Median Score: {median_score:.4f}
  • Std Dev:      {std_score:.4f}
  • Min Score:    {scores.min():.4f} (Shot {scores.idxmin()})
  • Max Score:    {outlier_score:.4f} (Shot {outlier_id})

ALL SCORES (SORTED):
{'─'*70}
"""
    
    for shot_id, score in sorted_scores.items():
        marker = " <-- OUTLIER" if shot_id == outlier_id else ""
        stats_text += f"  Shot {shot_id:2d}: {score:.4f}{marker}\n"
    
    stats_text += f"""
{'─'*70}
ALGORITHM: L2 Norm of Z-scores across 39 extracted features
CONCLUSION: Shot {outlier_id} is clearly anomalous

GENERATED: {Path('results').absolute()}
"""
    
    # Save statistics to text file
    stats_filename = 'results/outlier_statistics.txt'
    with open(stats_filename, 'w', encoding='utf-8') as f:
        f.write(stats_text)
    print(f'Saved {stats_filename}')
    
    print(f'\nOutlier detected: Shot {outlier_id}')
    print(f'Outlier score: {outlier_score:.4f}')
    print(f'  - {pct_above_second:.1f}% higher than 2nd highest')
    print(f'  - {std_deviations:.2f} standard deviations above mean')


def plot_outlier_verification_combined():
    """Generate combined outlier verification plot showing all metrics in one image."""
    # load data and compute outlier
    shots = load_shots()
    aligned, stats = align_and_normalize(shots)
    features = compute_features(aligned)
    scores = compute_outlier_scores(features)
    outlier_id = find_most_anomalous_shot(scores)

    # key metrics for verification - ALL 6 metrics
    metrics = [
        ('espresso_pressure', 'Pressure (bar)'),
        ('espresso_flow', 'Flow (ml/s)'),
        ('espresso_temperature_basket', 'Temperature Basket (°C)'),
        ('espresso_temperature_mix', 'Temperature Mix (°C)'),
        ('espresso_water_dispensed', 'Water Dispensed (ml)'),
        ('espresso_resistance', 'Resistance'),
    ]

    # create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (metric_name, ylabel) in enumerate(metrics):
        ax = axes[idx]

        # check if metric exists
        if metric_name not in shots[1].columns:
            ax.text(0.5, 0.5, f'{ylabel}\n(Not Available)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(ylabel)
            continue

        # plot all non-outlier shots in blue/light blue
        for shot_id, df in shots.items():
            if shot_id == outlier_id:
                continue
            ax.plot(df['espresso_elapsed'], df[metric_name],
                   color='#6495ED', alpha=0.4, linewidth=1.2)

        # plot outlier in bold red on top
        df_outlier = shots[outlier_id]
        ax.plot(df_outlier['espresso_elapsed'], df_outlier[metric_name],
               color='#DC143C', linewidth=2.5, label=f'Shot {outlier_id} (OUTLIER)',
               zorder=10)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    plt.suptitle(f'Outlier Verification: Shot {outlier_id} vs All Other Shots', fontsize=16, y=0.995)
    plt.tight_layout()

    # save combined figure
    filename = 'results/outlier_verification.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f'Saved {filename}')


if __name__ == "__main__":
    # ensure results directory exists
    Path('results').mkdir(exist_ok=True)
    
    print("Generating individual profile plots...")
    plot_all_profiles()
    
    print("\nGenerating combined profiles plot...")
    plot_all_profiles_combined()
    
    print("\nGenerating outlier verification plots (individual)...")
    plot_outlier_verification_individual()

    print("\nGenerating combined outlier verification plot...")
    plot_outlier_verification_combined()

    print("\n[DONE] All visualizations complete!")

