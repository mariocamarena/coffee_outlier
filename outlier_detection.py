import numpy as np
import pandas as pd

from data_loader import load_shots
from preprocess import align_and_normalize
from features import compute_features


def compute_outlier_scores(features: pd.DataFrame) -> pd.Series:
    """Compute outlier score for each shot using L2 norm of z-scores."""
    # standardize each feature
    z_scores = (features - features.mean()) / features.std()
    # print(f"z_scores shape: {z_scores.shape}")  # debug

    # compute L2 norm across features for each shot
    scores = np.sqrt((z_scores ** 2).sum(axis=1))

    return scores

# def compute_outlier_scores(features: pd.DataFrame) -> pd.Series:
#     eps = np.finfo(float).eps
#     med = features.median()
#     mad = (features - med).abs().median().replace(0, eps)
#     z = 0.6745 * (features - med) / mad  # robust z
#     scores = np.sqrt((z ** 2).sum(axis=1))
#     return scores


def find_most_anomalous_shot(scores: pd.Series) -> int:
    """Return shot id with highest outlier score."""
    return scores.idxmax()


if __name__ == "__main__":
    shots = load_shots()
    aligned, stats = align_and_normalize(shots)
    features = compute_features(aligned)
    scores = compute_outlier_scores(features)
    
    print("Outlier scores:")
    print(scores.sort_values(ascending=False))
    print(f"\nMost anomalous shot: {find_most_anomalous_shot(scores)}")

