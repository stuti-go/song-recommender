import pandas as pd  # Import pandas to handle DataFrame operations
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to evaluate the recommender system using Precision, Recall, and F1-Score
def evaluate_recommender_system(true_relevant_tracks, recommended_tracks):
    # Assuming 'true_relevant_tracks' and 'recommended_tracks' are DataFrames with track names and artists
    merged_df = pd.merge(recommended_tracks, true_relevant_tracks, on=['track_name', 'artists'], how='left')

    # If a track from the recommended list is found in the ground truth, it's relevant (1), otherwise not (0)
    merged_df['relevant'] = merged_df['relevant'].fillna(0)  # Filling missing relevance as 0

    # True positives (relevant and recommended), false positives (recommended but not relevant)
    y_true = merged_df['relevant']  # Ground truth values (whether the track is relevant)
    y_pred = [1] * len(recommended_tracks)  # All recommended tracks are considered as predicted positive

    # Calculate precision, recall, and F1-score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1
