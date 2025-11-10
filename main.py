from data_loader import load_shots
from preprocess import align_and_normalize
from features import compute_features
from outlier_detection import compute_outlier_scores, find_most_anomalous_shot


if __name__ == "__main__":
    # Run full pipeline
    print("Loading shots...")
    shots = load_shots()

    print("Aligning and normalizing...")
    aligned, stats = align_and_normalize(shots)

    print("Computing features...")
    features = compute_features(aligned)

    print("Computing outlier scores...")
    scores = compute_outlier_scores(features)

    # Find outlier
    outlier_shot = find_most_anomalous_shot(scores)

    # Display results
    print("\n" + "="*50)
    print("OUTLIER SCORES (sorted)")
    print("="*50)
    scores_sorted = scores.sort_values(ascending=False)
    for shot_id, score in scores_sorted.items():
        marker = " <-- OUTLIER" if shot_id == outlier_shot else ""
        print(f"Shot {shot_id:2d}: {score:.4f}{marker}")
    
    print("\n" + "="*50)
    print(f"Predicted bad cup: {outlier_shot}")
    print("="*50)

    # Save results
    scores.to_csv('results/outlier_scores.csv', header=['outlier_score'])
    print("\nResults saved to results/outlier_scores.csv")

