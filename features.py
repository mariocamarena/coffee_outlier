import numpy as np
import pandas as pd

from data_loader import load_shots
from preprocess import align_and_normalize


def compute_features(aligned_shots: dict) -> pd.DataFrame:
    """Extract features from aligned shots."""
    features = []

    for shot_id, df in aligned_shots.items():
        shot_features = {'shot_id': shot_id}

        # Compute mean, max, sum for each column
        for col in df.columns:
            if col != 'espresso_elapsed' and pd.api.types.is_numeric_dtype(df[col]):
                shot_features[f'{col}_mean'] = df[col].mean()
                shot_features[f'{col}_max'] = df[col].max()
                shot_features[f'{col}_sum'] = df[col].sum()
        
        features.append(shot_features)
    
    return pd.DataFrame(features).set_index('shot_id')


if __name__ == "__main__":
    shots = load_shots()
    aligned, stats = align_and_normalize(shots)
    features_df = compute_features(aligned)
    
    print(f"Feature matrix shape: {features_df.shape}")
    print(f"\nFirst 3 shots:")
    print(features_df.head(3))

