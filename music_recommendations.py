import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the datasets
music_data = pd.read_csv("data/data.csv")
genre_data = pd.read_csv('data/data_by_genres.csv')
year_data = pd.read_csv('data/data_by_year.csv')
artist_data = pd.read_csv('data/data_by_artist.csv')

# Preprocessing
music_data['decade'] = pd.to_datetime(music_data['year'], format='%Y').dt.year // 10 * 10
music_data['decade'] = pd.Categorical(music_data['decade'])
music_data['release_date'] = pd.to_datetime(music_data['release_date'], errors='coerce')
music_data['release_decade'] = (music_data['release_date'].dt.year // 10) * 10

# Check for missing values and duplicates
print("Missing values in music_data:\n", music_data.isnull().sum())
print("Duplicate rows in music_data:\n", music_data[music_data.duplicated()])

# List of numerical columns to consider for similarity calculations
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

# Normalize and scale the song data
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Prepare scaled and normalized data
normalized_data = min_max_scaler.fit_transform(music_data[number_cols])
scaled_normalized_data = standard_scaler.fit_transform(normalized_data)

# Function to retrieve song data for a given song name
def get_song_data(name, data):
    try:
        song_data = data[data['name'].str.lower() == name.lower()].iloc[0]
        return song_data
    except IndexError:
        return None

# Function to calculate the mean vector of a list of songs
def get_mean_vector(song_list, data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song['name'], data)
        if song_data is None:
            print('Warning: {} does not exist in the dataset'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)
    if not song_vectors:
        return None
    song_matrix = np.array(song_vectors)
    return np.mean(song_matrix, axis=0)

# Function to recommend songs based on a list of seed songs
def recommend_songs(seed_songs, data, n_recommendations=10):
    metadata_cols = ['name', 'artists', 'year']
    song_center = get_mean_vector(seed_songs, data)
    
    # Return an empty list if song_center is missing
    if song_center is None:
        return []
    
    # Normalize the song center
    normalized_song_center = min_max_scaler.transform([song_center])
    
    # Standardize the normalized song center
    scaled_normalized_song_center = standard_scaler.transform(normalized_song_center)
    
    # Calculate cosine similarities
    similarities = cosine_similarity(scaled_normalized_song_center, scaled_normalized_data)[0]
    
    # Get indices of songs with the highest similarities
    indices = np.argsort(similarities)[-n_recommendations:][::-1]
    
    # Filter out seed songs and duplicates, then get the top n_recommendations
    rec_songs = []
    for i in indices:
        song_name = data.iloc[i]['name']
        if song_name.lower() not in [song['name'].lower() for song in seed_songs]:
            rec_songs.append(data.iloc[i])
            if len(rec_songs) == n_recommendations:
                break
    
    return pd.DataFrame(rec_songs)[metadata_cols].to_dict(orient='records')

# Streamlit Application
st.title("Music Recommendation System")

# Streamlit sidebar for selecting actions
action = st.sidebar.selectbox("Choose Action", ["Visualize Data", "Recommend Songs"])

if action == "Visualize Data":
    st.write("### Data Visualization")
    
    # Visualize Decade Distribution
    st.write("#### Number of Songs Released per Decade")
    plt.figure(figsize=(11, 6))
    sns.countplot(data=music_data, x='decade')
    st.pyplot(plt)

    # Popularity Trends Over Years
    st.write("#### Popularity Trends Over the Years")
    popularity_fig = px.line(year_data, x='year', y='popularity', title='Popularity Trends Over the Years', labels={'year': 'Years', 'popularity': 'Popularity'})
    st.plotly_chart(popularity_fig)

    # Top Genres by Popularity
    st.write("#### Top Genres by Popularity")
    top_10_genre_data = genre_data.nlargest(10, 'popularity')
    genre_fig = px.bar(top_10_genre_data, x='popularity', y='genres', orientation='h', title='Top Genres by Popularity', color='genres', labels={'popularity': 'Popularity', 'genres': 'Genres'})
    st.plotly_chart(genre_fig)

    # Top Artists by Popularity
    st.write("#### Top Artists by Popularity")
    top_10_artist_data = artist_data.nlargest(10, 'popularity')
    artist_fig = px.bar(top_10_artist_data, x='popularity', y='artists', orientation='h', title='Top Artists by Popularity', color='artists', labels={'popularity': 'Popularity', 'artists': 'Artists'})
    st.plotly_chart(artist_fig)

    # Top Songs by Popularity
    st.write("#### Top Songs by Popularity")
    top_songs = music_data.nlargest(10, 'popularity')
    song_fig = px.bar(top_songs, x='popularity', y='name', orientation='h', title='Top Songs by Popularity', color='name', labels={'popularity': 'Popularity', 'name': 'Name'})
    st.plotly_chart(song_fig)

elif action == "Recommend Songs":
    st.write("### Recommend Songs")
    seed_songs = st.text_area("Enter seed songs (comma-separated):")
    seed_songs_list = [{'name': song.strip()} for song in seed_songs.split(',')]
    
    n_recommendations = st.number_input("Number of recommendations", min_value=1, max_value=20, value=10)
    
    if st.button("Get Recommendations"):
        recommended_songs = recommend_songs(seed_songs_list, music_data, n_recommendations)
        
        if recommended_songs:
            st.write("### Recommended Songs")
            for idx, song in enumerate(recommended_songs, start=1):
                st.write(f"{idx}. {song['name']} by {song['artists']} ({song['year']})")
        else:
            st.write("No recommendations found. Please check the seed songs and try again.")


#streamlit run music_recommendations.py
