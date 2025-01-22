import pandas as pd
from musicrec import recommend_tracks_by_name_artist, recommend_tracks_by_cluster  # Import the recommendation function
from evaluation import evaluate_recommender_system  # Import the evaluation function

# Manually input the top_n relevant tracks (ground truth)
true_relevant_tracks = pd.DataFrame({
    'track_name': ['Crawling', 'In The End', 'Softcore', 'A Place for My Head'],
    'artists': ['Linkin Park', 'Linkin Park', 'The Neighbourhood', 'Linkin Park'],
    'track_genre': ['alternative', 'alternative', 'alt-rock', 'alternative'],
    'relevant': [1, 1, 1, 1]  # True relevant songs marked as 1
})

# Input for the track to recommend and the number of recommendations
track_name = 'Numb'
artist_name = 'Linkin Park'
top_n = 5  # Number of recommendations to generate

# Generate recommendations using musicrec.py
recommended_tracks = recommend_tracks_by_name_artist(track_name, artist_name, top_n=top_n)
print(recommended_tracks)

# Evaluate the recommendations using evaluation.py
precision, recall, f1 = evaluate_recommender_system(true_relevant_tracks, recommended_tracks)

# Print the evaluation metrics
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

print(" ")

# Generate recommendations using musicrec.py
recommended_tracks = recommend_tracks_by_cluster(track_name, artist_name, top_n=top_n)
print(recommended_tracks)

# Evaluate the recommendations using evaluation.py
precision, recall, f1 = evaluate_recommender_system(true_relevant_tracks, recommended_tracks)

# Print the evaluation metrics
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
