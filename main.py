from flask import Flask, render_template, request
import requests
import re
import spotipy
import csv
from spotipy.oauth2 import SpotifyClientCredentials
import random
import string

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')




@app.route('/get_songs', methods=['POST'])
def get_songs():
    # Get the playlist URL from the form on the webpage
    playlist_url = request.form['playlist_url']

    # Extract the playlist ID from the URL
    match = re.search('(https://open.spotify.com/playlist/)(\w+)(\?.*|$)', playlist_url)
    playlist_id = match.group(2)

    # Get an access token for the Spotify Web API
    client_id = '59716ee804104ee7b913e244cc67fd55'
    client_secret = 'ed17be9c7e0d44838de69ec300b36849'
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_response = requests.post(auth_url, {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
    })
    auth_response_data = auth_response.json()
    access_token = auth_response_data['access_token']

    # Use the access token to get the playlist data
    api_url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(api_url, headers=headers)
    response_data = response.json()

    # Get the song names and audio features from the playlist data
    song_names = []
    audio_features = []
    sp = spotipy.Spotify(auth=access_token)
    for item in response_data['items']:
        track_uri = item['track']['uri']
        track_name = item['track']['name']
        song_names.append(track_name)
        audio_features.append(sp.audio_features(track_uri)[0])

    # Write the song data to a CSV file
    with open('songs.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration (ms)'])
        for name, features in zip(song_names, audio_features):
            writer.writerow([name, features['danceability'], features['energy'], features['key'], features['loudness'], features['mode'], features['speechiness'], features['acousticness'], features['instrumentalness'], features['liveness'], features['valence'], features['tempo'], features['duration_ms']])

    # Now we get a random track.
    random_search = '%25' + random.choice(string.ascii_letters) + '%25'
    random_offset = random.randint(0, 1000)
    random_track_url = f'https://api.spotify.com/v1/search?q={random_search}&type=track&market=US&limit=1&offset={random_offset}'
    random_response = requests.get(random_track_url, headers=headers).json()['tracks']['items'][0]
    random_name = random_response['name']
    random_features = sp.audio_features(random_response['uri'])[0]

    with open('rand_song.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness', \
                'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration (ms)'])
        writer.writerow([random_name, random_features['danceability'], random_features['energy'], random_features['key'], random_features['loudness'], \
                random_features['mode'], random_features['speechiness'], random_features['acousticness'], random_features['instrumentalness'], \
                random_features['liveness'], random_features['valence'], random_features['tempo'], random_features['duration_ms']])

    # Render the song data on a new webpage
    return render_template('songs.html', song_names=song_names, audio_features=audio_features, zip=zip, \
            random_name=random_name, random_features=random_features)

if __name__ == '__main__':
    app.run(debug=True)
