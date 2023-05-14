from flask import Flask, render_template, request
import requests
import re
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
    # Get the song names from the playlist data
    song_names = []
    for item in response_data['items']:
        song_names.append(item['track']['name'])

    # Render the song names on a new webpage
    return render_template('songs.html', song_names=song_names)


if __name__ == '__main__':
    app.run(debug=True)
