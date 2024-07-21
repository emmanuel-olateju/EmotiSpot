import sys
import json
import threading
import joblib
import numpy as np
from scipy.signal import resample
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from pylsl import StreamInlet, resolve_stream
import torch
import google.generativeai as genai
from modules.neural_nets import *

# Spotify API credentials (replace with your actual credentials)
CLIENT_ID = "bf6878ee234b45c69d67397512da4c24"
CLIENT_SECRET = "d883a0a24086473992f15f090db8f620"
REDIRECT_URI = "https://localhost:8888/callback"
SCOPE = 'user-library-read user-modify-playback-state user-read-playback-state'

GOOGLE_API_KEY = "AIzaSyCAA7XNVJ7GnXFajT0rVRLBf9Pt872yYL0"

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.sp = self.authenticate_spotify()
        self.emotion = None
        self.song = None
        self.eeg_thread = None
        self.eeg_running = False
        self.start_eeg_thread()

    def authenticate_spotify(self):
        auth_manager = SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope=SCOPE)
        return Spotify(auth_manager=auth_manager)

    def initUI(self):
        self.setWindowTitle('Emotion Selector')
        self.setGeometry(100, 100, 400, 300)

        # Set black background
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(Qt.black))
        self.setPalette(palette)

        # Create layout
        layout = QVBoxLayout()
        self.emotion_label = QLabel('', self)
        self.emotion_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.emotion_label)

        # Set main layout
        self.setLayout(layout)
        self.show()
    
    def closeEvent(self, event):
        if self.eeg_running==True:
            self.stop_eeg_thread()
        event.accept()

    def search_and_play_spotify_song(self, query):
        if self.sp is None:
            self.sp = self.authenticate_spotify()
        
        results = self.sp.search(q=query, type='track', limit=1)
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            track_uri = track['uri']
            print(f"Found song: {track['name']} by {track['artists'][0]['name']}")
            print(track_uri)
            self.play_song(track_uri)
        else:
            print("No song found")

    def play_song(self, track_uri):
        devices = self.sp.devices()
        if devices['devices']:
            device_id = devices['devices'][0]['id']
            self.sp.start_playback(device_id=device_id, uris=[track_uri])
            print("Playing song")
        else:
            print("No active devices found")

    def pause_playback(self):
        try:
            self.sp.pause_playback()
            print("Playback paused")
        except Exception as e:
            print(f"Error: {e}")

    def start_eeg_thread(self):
        self.eeg_running = True
        self.eeg_thread = threading.Thread(target=self.eeg_classification)
        self.eeg_thread.start()

    def stop_eeg_thread(self):
        self.eeg_running = False
        if self.eeg_thread is not None:
            self.eeg_thread.join()

    def eeg_classification(self):
        print("Looking for an EEG stream...")
        streams = resolve_stream('type', 'EEG')
        print("Found EEG Stream")

        inlet = StreamInlet(streams[0])

        num_samples = 1280
        num_input_channels = 4
        num_output_channels = 14

        eeg_data = np.zeros((num_samples, num_input_channels))

        est_model = joblib.load("artifacts/models/cnn_22.sav")
        est_model = est_model["model"]
        est_model = est_model.eval()

        clf_model = joblib.load("artifacts/models/cnn_23_clf.sav")
        clf_model = clf_model["model"]
        clf_model = clf_model.eval()

        while self.eeg_running:
            print("Collecting EEG data...")
            for i in range(num_samples):
                sample, timestamp = inlet.pull_sample()
                eeg_data[i, :] = sample[:num_input_channels]

            upscaled_eeg_data = self.upscale_channels(eeg_data, num_output_channels)

            input_data = upscaled_eeg_data.reshape(1, num_samples, num_output_channels, 1)
            input_data = np.squeeze(input_data, axis=-1)
            input_data = torch.tensor(input_data).to(torch.float)

            valence_arousal = est_model(input_data)
            prediction = clf_model(input_data)

            emotion = self.get_emotion(prediction)
            valence, arousal = valence_arousal[0,0].item(), valence_arousal[0,1].item()

            self.update_emotion_label(emotion, valence, arousal)

    def upscale_channels(self, data, num_output_channels):
        num_samples, num_input_channels = data.shape
        upscaled_data = np.zeros((num_samples, num_output_channels))
        
        for i in range(num_output_channels):
            original_channel = i % num_input_channels
            upscaled_data[:, i] = data[:, original_channel] * (1 + 0.1 * (i // num_input_channels))
            
        return upscaled_data

    def get_emotion(self, prediction):
        # This function maps the model output to an emotion
        emotions = ['Happy', 'Sad', 'Calm', 'Angry']
        return emotions[prediction.argmax().item()]

    def update_emotion_label(self, emotion, valence, arousal):
        self.emotion_label.setText(f"Emotion: {emotion}")

        print(emotion, self.emotion)
        if (self.emotion!=emotion) or (self.song==None):
            self.emotion = emotion
            self.song = self.get_song_recommendation(emotion)
            print(self.song)
            if self.song:
                self.search_and_play_spotify_song(self.song)
        else:
            print("Song in Play")

        # Update the text color based on valence and arousal
        if emotion == 'Happy':
            color = QColor(0, 255, 0) if valence > 0.5 and arousal > 0.5 else QColor(0, 128, 0)
        elif emotion == 'Sad':
            color = QColor(0, 0, 255) if valence < 0.5 and arousal < 0.5 else QColor(0, 0, 128)
        elif emotion == 'Calm':
            color = QColor(255, 255, 255) if valence > 0.5 and arousal < 0.5 else QColor(128, 128, 128)
        elif emotion == 'Angry':
            color = QColor(255, 0, 0) if valence < 0.5 and arousal > 0.5 else QColor(128, 0, 0)
        else:
            color = QColor(255, 255, 255)

        palette = self.emotion_label.palette()
        palette.setColor(QPalette.WindowText, color)
        self.emotion_label.setPalette(palette)

    def get_song_recommendation(self, emotion):
        # Replace this URL with the actual API endpoint

        try:
            prompt = f"Give me the name of just one song and its artist that fits the emotion {emotion}"
            response = model.generate_content(prompt).text
            return response
        except Exception as e:
            print(f"Error: {e}")
        return None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = EmotionApp()
    sys.exit(app.exec_())
