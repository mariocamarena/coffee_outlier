import numpy as np
import pandas as pd

from data_loader import load_shots


def resample_shot(df, num_points=200):
    """Resample shot to fixed number of points."""
    # use elapsed time for interpolation
    # TODO: could try cubic spline instead of linear interp
    if 'espresso_elapsed' in df.columns:
        time_col = df['espresso_elapsed']
        # Create evenly spaced time points
        new_time = np.linspace(time_col.min(), time_col.max(), num_points)

        # interpolate each column
        resampled = {'espresso_elapsed': new_time}
        for col in df.columns:
            if col != 'espresso_elapsed' and pd.api.types.is_numeric_dtype(df[col]):
                resampled[col] = np.interp(new_time, time_col, df[col])
        
        return pd.DataFrame(resampled)
    else:
        # Fallback: use index
        old_idx = np.arange(len(df))
        new_idx = np.linspace(0, len(df) - 1, num_points)
        
        resampled = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                resampled[col] = np.interp(new_idx, old_idx, df[col])
        
        return pd.DataFrame(resampled)


def align_and_normalize(shots: dict, num_points=200):
    """Align all shots to same length and z-score normalize."""
    # Resample all shots
    aligned = {}
    for shot_id, df in shots.items():
        aligned[shot_id] = resample_shot(df, num_points)

    # collect all data for normalization
    all_data = pd.concat(aligned.values(), ignore_index=True)

    # compute stats for z-score
    stats = {}
    for col in all_data.columns:
        if col != 'espresso_elapsed':  # don't normalize time
            stats[col] = {
                'mean': all_data[col].mean(),
                'std': all_data[col].std()
            }

    # apply z-score
    for shot_id in aligned:
        for col, stat in stats.items():
            if col in aligned[shot_id].columns:
                aligned[shot_id][col] = (aligned[shot_id][col] - stat['mean']) / stat['std']
    
    return aligned, stats


if __name__ == "__main__":
    shots = load_shots()
    aligned, stats = align_and_normalize(shots)

    # Check result
    first = aligned[1]
    print(f"Resampled shape: {first.shape}")
    print(f"Columns: {list(first.columns[:5])}...")
    print(f"Normalized {len(stats)} features")

