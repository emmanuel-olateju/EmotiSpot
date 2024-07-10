import sys
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from pylsl import StreamInlet, resolve_stream
import numpy as np

# Spotify API credentials (replace with your actual credentials)
CLIENT_ID = "bf6878ee234b45c69d67397512da4c24"
CLIENT_SECRET = "d883a0a24086473992f15f090db8f620"
REDIRECT_URI = "https://localhost:8888/callback"
SCOPE = 'user-library-read user-modify-playback-state user-read-playback-state'

class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.sp = self.authenticate_spotify()

    def authenticate_spotify(self):
        auth_manager = SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope=SCOPE)
        return Spotify(auth_manager=auth_manager)

    def initUI(self):
        self.setWindowTitle('Emotion Selector')

        # Create layout
        layout = QVBoxLayout()

        # Create text inputs
        self.sad_input = QLineEdit(self)
        self.anger_input = QLineEdit(self)
        self.happy_input = QLineEdit(self)
        self.calm_input = QLineEdit(self)

        # Add text inputs to layout
        layout.addLayout(self.create_input_layout('Sad', self.sad_input))
        layout.addLayout(self.create_input_layout('Anger', self.anger_input))
        layout.addLayout(self.create_input_layout('Happy', self.happy_input))
        layout.addLayout(self.create_input_layout('Calm', self.calm_input))

        # Create dropdown
        self.song_selector = QComboBox(self)
        self.song_selector.addItems(['Sad', 'Anger', 'Happy', 'Calm'])
        layout.addWidget(QLabel('Select Mood:'))
        layout.addWidget(self.song_selector)

        # File Directory/Name
        self.file = QLineEdit(self)
        layout.addLayout(self.create_input_layout('File', self.file))

        # Create buttons
        self.start_button = QPushButton('Start', self)
        self.stop_button = QPushButton('Stop', self)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)

        self.EEG_recording_start = QPushButton('Start EEG Recording', self)
        self.EEG_recording_stop = QPushButton('Stop EEG Recording', self)
        EEG_recording_layout = QHBoxLayout()
        EEG_recording_layout.addWidget(self.EEG_recording_start)
        EEG_recording_layout.addWidget(self.EEG_recording_stop)
        layout.addLayout(EEG_recording_layout)

        # Connect Buttons to Methods
        self.start_button.clicked.connect(self.on_start)
        self.stop_button.clicked.connect(self.on_stop)
        self.EEG_recording_start.clicked.connect(self.record)
        self.EEG_recording_stop.clicked.connect(self.stop_record)

        # Set main layout
        self.setLayout(layout)
        self.show()

    def create_input_layout(self, label_text, line_edit):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text + ':'))
        layout.addWidget(line_edit)
        return layout
    
    def on_start(self):
        self.sad_value = self.sad_input.text()
        self.anger_value = self.anger_input.text()
        self.happy_value = self.happy_input.text()
        self.calm_value = self.calm_input.text()
        self.selected_mood = self.song_selector.currentText()

        # Print values to console (or handle them as needed)
        print(f"Sad: {self.sad_value}")
        print(f"Anger: {self.anger_value}")
        print(f"Happy: {self.happy_value}")
        print(f"Calm: {self.calm_value}")
        print(f"Selected Song: {self.selected_mood}")

        # Handle the logic for the selected song
        if self.selected_mood == 'Calm':
            # Run the Spotify search in a separate thread to avoid blocking the GUI
            threading.Thread(target=self.search_and_play_spotify_song, args=(self.calm_value,)).start()
        elif self.selected_mood == 'Sad':
            # Run the Spotify search in a separate thread to avoid blocking the GUI
            threading.Thread(target=self.search_and_play_spotify_song, args=(self.sad_value,)).start()
        if self.selected_mood == 'Anger':
            # Run the Spotify search in a separate thread to avoid blocking the GUI
            threading.Thread(target=self.search_and_play_spotify_song, args=(self.anger_value,)).start()
        if self.selected_mood == 'Happy':
            # Run the Spotify search in a separate thread to avoid blocking the GUI
            threading.Thread(target=self.search_and_play_spotify_song, args=(self.happy_value,)).start()

    def on_stop(self):
        # Stop the music playback
        self.pause_playback()

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

    def record(self):
        threading.Thread(target=self.record_eeg_stream).start()

    def record_eeg_stream(self):
        # Resolve the EEG stream
        print("Looking for an EEG stream...")
        streams = resolve_stream('type', 'EEG')
        if not streams:
            print("No EEG streams found.")
            return
        else:
            print(streams[0])

        # Open an inlet to the EEG stream
        inlet = StreamInlet(streams[0])
        stream_info = inlet.info()
        channel_count = stream_info.channel_count()
        self.is_recording = True

        data = np.empty((0,channel_count))

        # Start recording EEG data
        print("Recording EEG stream...")
        while self.is_recording:
            sample, timestamp = inlet.pull_sample()
            data = np.vstack((data,sample))
            print(f"Data Size: {data.shape}")

        np.save(self.file.text(), data)
        print("Stopped recording EEG stream.")
    
    def stop_record(self):
        self.is_recording = False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = EmotionApp()
    sys.exit(app.exec_())
