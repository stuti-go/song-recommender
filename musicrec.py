import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.cluster import KMeans

# Read the dataset
df = pd.read_csv(r"D:\Stuti\OneDrive\Desktop\college\ML\music recommender system\dataset\dataset.csv")

# Preprocess the dataset: Dropping duplicates and handling missing values
df = df.drop_duplicates(subset=['track_name', 'artists'])
df = df.dropna(subset=['track_name', 'artists', 'danceability', 'energy', 'tempo', 'loudness', 'valence', 'liveness', 'track_genre', 'popularity'])

# Ensure genre is categorical and handle it accordingly
df['track_genre'] = df['track_genre'].fillna('Unknown')  # Handle missing genres if any

# Encode genre using OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
genre_encoded = encoder.fit_transform(df[['track_genre']])

# Normalize the numerical features (this is important for KNN and cosine similarity)
features = df[['danceability', 'energy', 'tempo', 'loudness', 'valence', 'liveness']].values
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Combine the scaled numerical features with the encoded genre features
combined_features = np.hstack([scaled_features, genre_encoded])

# Initialize KNN
knn = NearestNeighbors(n_neighbors=100, algorithm='ball_tree', metric='euclidean')
knn.fit(combined_features)

# Function to find track index by name and artist
def find_track_index(track_name, artist_name):
    track_row = df[(df['track_name'] == track_name) & (df['artists'] == artist_name)]
    if track_row.empty:
        raise ValueError(f"Track '{track_name}' by {artist_name} not found in the dataset.")
    return track_row.index[0]

# Function to recommend tracks based on track name and artist
def recommend_tracks_by_name_artist(track_name, artist_name, top_n=5, popularity_threshold=50):
    # Find the track index based on track name and artist
    try:
        track_index = find_track_index(track_name, artist_name)
    except ValueError as e:
        print(e)
        return None

    # Filter the dataset to only include popular tracks (above a certain popularity threshold)
    df_popular = df[df['popularity'] >= popularity_threshold]

    # Ensure the input track is included, even if it's below the popularity threshold
    input_track = df[(df['track_name'] == track_name) & (df['artists'] == artist_name)]
    df_popular = pd.concat([df_popular, input_track]).drop_duplicates(subset=['track_name', 'artists'])

    # Get the nearest neighbors using KNN for the filtered dataset
    filtered_features = df_popular[['danceability', 'energy', 'tempo', 'loudness', 'valence', 'liveness']].values
    scaled_filtered_features = scaler.transform(filtered_features)  # Scale using the same scaler
    combined_filtered_features = np.hstack([scaled_filtered_features, encoder.transform(df_popular[['track_genre']])])
    
    knn.fit(combined_filtered_features)
    distances, indices = knn.kneighbors(combined_filtered_features[df_popular.index.get_loc(track_index)].reshape(1, -1))

    # Handle cases where there are not enough neighbors
    num_recommendations = min(top_n, len(indices[0]) - 1)  # Don't count the input track itself
    recommended_tracks = df_popular.iloc[indices[0][1:num_recommendations+1]]  # Skip the first as it's the input track itself

    # Return the recommended tracks, including track name, artist, genre, and popularity
    recommended_tracks = recommended_tracks[['track_name', 'artists', 'track_genre']]
    return recommended_tracks

# Test the recommendation system with the song 'Numb' by 'Linkin Park'
# recommended_tracks = recommend_tracks_by_name_artist('Numb', 'Linkin Park', top_n=10, popularity_threshold=50)
# print(recommended_tracks)

genre_weight = 10  # Adjust this factor to control the emphasis on genre
scaled_genre = genre_encoded * genre_weight

# Combine the numerical and weighted genre features
combined_features = np.hstack([scaled_features, scaled_genre])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans.fit_predict(combined_features)

# Function to recommend tracks based on KMeans clustering and popularity threshold
def recommend_tracks_by_cluster(track_name, artist_name, top_n=5, popularity_threshold=50):
    # Filter the dataset to only include tracks with popularity above the threshold
    df_filtered = df[df['popularity'] >= popularity_threshold]
    
    # Find the cluster of the input song
    track_row = df_filtered[(df_filtered['track_name'] == track_name) & (df_filtered['artists'] == artist_name)]
    if track_row.empty:
        raise ValueError(f"Track '{track_name}' by {artist_name} not found in the dataset.")
    
    input_cluster = track_row['cluster'].values[0]
    
    # Get all songs in the same cluster
    recommended_tracks = df_filtered[df_filtered['cluster'] == input_cluster]
    
    # Exclude the input song itself
    recommended_tracks = recommended_tracks[recommended_tracks['track_name'] != track_name]
    
    # Return top N recommendations
    return recommended_tracks[['track_name', 'artists', 'track_genre']].head(top_n)

# Test the recommendation system with the song 'Numb' by 'Linkin Park' and a popularity threshold of 50
recommended_tracks = recommend_tracks_by_name_artist('Toxicity', 'System Of A Down', top_n=10, popularity_threshold=50)
print(recommended_tracks)