from flask import Flask, request, redirect
from spotipy.oauth2 import SpotifyOAuth
import json

app = Flask(__name__)

# Spotify API credentials (replace with your actual credentials)
CLIENT_ID = "bf6878ee234b45c69d67397512da4c24"
CLIENT_SECRET = "d883a0a24086473992f15f090db8f620"
REDIRECT_URI = 'https://localhost:8888/callback'
SCOPE = 'user-library-read user-modify-playback-state user-read-playback-state'

# Set up Spotify OAuth
sp_oauth = SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope='user-library-read')

@app.route('/')
def index():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    return "Spotify authentication successful. You can now close this window."

if __name__ == '__main__':
    app.run(port=8888, ssl_context='adhoc')
