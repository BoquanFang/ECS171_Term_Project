import os

import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import requests
import re
import spotipy
import csv
from spotipy.oauth2 import SpotifyClientCredentials
import random
import string
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas
import numpy
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
import math
from scipy.stats import norm
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
        writer.writerow(['Name', 'Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness', 'Acousticness',
                         'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration (ms)'])
        for name, features in zip(song_names, audio_features):
            writer.writerow([name, features['danceability'], features['energy'], features['key'], features['loudness'],
                             features['mode'], features['speechiness'], features['acousticness'],
                             features['instrumentalness'], features['liveness'], features['valence'], features['tempo'],
                             features['duration_ms']])

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
        writer.writerow(
            [random_name, random_features['danceability'], random_features['energy'], random_features['key'],
             random_features['loudness'], \
             random_features['mode'], random_features['speechiness'], random_features['acousticness'],
             random_features['instrumentalness'], \
             random_features['liveness'], random_features['valence'], random_features['tempo'],
             random_features['duration_ms']])

    # Render the song data on a new webpage

    return render_template('songs.html', song_names=song_names, audio_features=audio_features, zip=zip,
                           random_name=random_name, random_features=random_features)


@app.route('/get_song_data', methods=['POST'])
def get_song_data():
    music_orig = pd.read_csv('songs.csv', encoding='ISO-8859-1')
    # this will be the scraped datase. We need to rename it appropriately later
    list(music_orig.columns)
    music = music_orig.drop(columns=['Name'], axis='columns').dropna()

    scaled_music = pd.DataFrame(normalize(music, axis=0), columns=music.columns)
    scaled_music.head()
    plt.figure(figsize=(12, 8))
    scaled_music_corr = scaled_music.corr()
    maskval = np.tril(scaled_music_corr)
    sns.heatmap(scaled_music_corr, annot=True, cmap='PRGn', mask=maskval, square=False)
    plt.savefig('static/heatmap1.png')
    plt.close()

    # Get the URL of the heatmap image
    heatmap_image_url1 = '/static/heatmap1.png'
    scaled_music_corr_abs = scaled_music_corr.abs()

    to_drop = list()

    for column in scaled_music_corr_abs.columns:
        if scaled_music_corr_abs.loc['Energy', column] < 0.3:
            to_drop.append(column)

    scaled_music['energy_category'] = round(scaled_music['Energy'] / 0.01, 0)
    scaled_music_dataset_first = scaled_music.drop(columns=to_drop, axis=1)
    scaled_music_dataset = scaled_music_dataset_first.drop('Energy', axis=1)
    scaled_music_dataset.head()
    sns.FacetGrid(scaled_music_dataset, height=5) \
        .map(sns.histplot, 'energy_category', stat="density") \
        .add_legend();
    plt.savefig('static/histplot.png')
    plt.close()

    # Get the URL of the heatmap image
    histplot = '/static/histplot.png'
    plt.figure(figsize=(12, 8))
    scaled_music_dataset_corr = scaled_music_dataset.corr()
    maskval_dataset = np.tril(scaled_music_dataset_corr)
    sns.heatmap(scaled_music_dataset_corr, annot=True, cmap='PRGn', mask=maskval_dataset)
    plt.savefig('static/heatmap2.png')
    plt.close()

    # Get the URL of the heatmap image
    heatmap_image_url2 = '/static/heatmap2.png'




    average_energy_level = scaled_music_dataset['energy_category'].mean()
    sd = scaled_music_dataset['energy_category'].std()
    print(average_energy_level, sd)

    scaled_music_dataset['energy_category'] = pd.Categorical(scaled_music_dataset['energy_category'])
    # train, test = train_test_split(scaled_music_dataset, test_size=(10/len(scaled_music_dataset['energy_category'])) , random_state=1)
    # train, test = train_test_split(scaled_music_dataset, test_size=0.2 , random_state=1)
    # test_size = 0.001 for our current dataset will change to 0.1 during the actual model for 10-fold validation.
    # train_size will also be removed.
    X_train = scaled_music_dataset.drop('energy_category', axis=1)
    y_train = scaled_music_dataset['energy_category']

    new_song = pd.read_csv('rand_song.csv', encoding='ISO-8859-1')
    new_song_name = new_song['Name'].iloc[0]
    new_song = new_song.drop(columns=['Name'], axis='columns').dropna()
    new_song = new_song.drop(columns=to_drop, axis=1)
    new_song = new_song.drop('Energy', axis=1)
    new_song.head()
    X_test = new_song


    i = int(round(math.sqrt(len(X_train['Loudness'])), 0))
    print(i)
    neigh = KNeighborsClassifier(n_neighbors=i)
    # The model is properly trained here
    neigh_fit = neigh.fit(X_train, y_train)
    pred = neigh.predict(X_test)
    accuracy_val = abs(int(pred) - average_energy_level)

    x = np.linspace(average_energy_level - 3 * sd, average_energy_level + 3 * sd, 100)
    plt.plot(x, norm.pdf(x, average_energy_level, sd), label='Normal Curve')
    plt.axvline(x=pred, color='red', linestyle='--', label='Predicted Energy Level')
    plt.axvline(x=average_energy_level - 1.645 * sd, color='green', label='Confidence interval low')
    plt.axvline(x=average_energy_level + 1.645 * sd, color='green', label='Confidence interval high')
    plt.title('Prediction on Normal Curve')
    plt.xlabel('Energy Level')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig('static/interval.png')
    plt.close()

    # Get the URL of the heatmap image
    interval_image = '/static/interval.png'

    # Now we get a random track.
    song_url = request.form['song_url']
    # Do the final comparison and give the conclusion
    fit_status = "Not a good fit"
    if accuracy_val < 1.645 * sd:
        fit_status = "Good fit"



    # Extract the song ID from the URL
    match = re.search('(https://open.spotify.com/track/)(\w+)(\?.*|$)', song_url)
    song_id = match.group(2)

    # Set up the Spotify API client
    client_id = '59716ee804104ee7b913e244cc67fd55'
    client_secret = 'ed17be9c7e0d44838de69ec300b36849'
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # Use the access token to get the song data
    response_data = sp.track(song_id)

    # Get the song name and audio features
    random_name = response_data['name']
    random_features = sp.audio_features(song_id)[0]

    with open('rand_song.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness',
                         'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration (ms)'])
        writer.writerow(
            [random_features['danceability'], random_features['energy'], random_features['key'],
             random_features['loudness'], \
             random_features['mode'], random_features['speechiness'], random_features['acousticness'],
             random_features['instrumentalness'], \
             random_features['liveness'], random_features['valence'], random_features['tempo'],
             random_features['duration_ms']])

    return render_template('song.html', random_name=random_name, random_features=random_features, heatmap_image_url1=heatmap_image_url1,
                           heatmap_image_url2=heatmap_image_url2, histplot=histplot, fit_status=fit_status,  interval_image = interval_image, new_song_name=random_name)





@app.route('/song_data', methods=['POST'])
def song_data():
    # Retrieve the entered song name from the form
    song_name = request.form['song_name']

    return redirect(url_for('song_data', song_name=song_name))


if __name__ == '__main__':
    app.run(debug=True)
