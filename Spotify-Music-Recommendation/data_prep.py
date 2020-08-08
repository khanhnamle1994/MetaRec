# Import packages
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re

# Data Pre-processing and Feature Engineering
songs = pd.read_csv('spotify_dataset.csv', header=None, skiprows=[0], usecols=[0, 1, 2, 3]).dropna()
songs.columns = ['user_id', 'artist_name', 'track_name', 'playlist_name']
print("Shape of dataframe:", songs.shape)


# Every artist can have only one version of the song, reduces unique songs from 2009210 to 1367548
def preprocess_track(track_name):
    """
    Preprocess the track
    :param track_name: name of the track
    :return: pre-processed track
    """
    track_name = track_name.lower()

    # everything between [], () is often live remaster etc.
    track_name = re.sub("[\(\[].*?[\)\]]", "", track_name)

    # Remove everything after '-' as they most likely are live, remastered, year etc identifiers
    track_name = track_name.split('-', 1)[0]
    # Remove empty spaces around the track_name
    track_name = track_name.strip()

    return track_name


songs['track_name'] = songs['track_name'].copy().apply(preprocess_track)
# Lowercase the artist names
songs['artist_name'] = songs['artist_name'].map(lambda x: x.lower())

print("Number of unique tracks:", songs['track_name'].nunique())

# Make two new features
# track_artist: identify the specific track of the artist to differ between same named songs from different artists
# user_playlist: identify the specific track to differ between same named playlists from different users
songs["track_artist"] = songs["artist_name"] + " - " + songs["track_name"]
songs["user_playlist"] = songs["user_id"] + " - " + songs["playlist_name"]
print("Number of unique songs when artist names are added to song name:", songs['track_artist'].nunique())

frequency_of_songs = songs.groupby('track_artist').count()['user_id']
sorted_frequencies = frequency_of_songs.sort_values(ascending=False)

# Visualize the rank vs frequency of tracks
plt.plot(np.arange(1, 2342854), sorted_frequencies[0:2342853])
plt.ylabel('frequency')
plt.xlabel('rank')
plt.xscale('log')
plt.savefig('Rank-vs-Frequency-of-Tracks')

# Remove playlist with less than 10 tracks and combine into lists
playlists = songs.groupby(['user_playlist']).filter(lambda x: len(x) >= 10)
playlists = playlists.groupby(['user_playlist']).agg({'track_artist': lambda x: list(x)})


def playlist_format(playlists):
    """
    Format the playlist
    :param playlists: a list of playlists
    :return: playlists formatted as documents
    """
    documents = []
    for index, row in playlists.iterrows():
        preprocessed_songs = row['track_artist']
        documents.append(preprocessed_songs)

    return documents


playlist_formatted = playlist_format(playlists)
playlist_length = len(playlist_formatted)
print("Total number of playlists:", playlist_length)
