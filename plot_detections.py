import argparse
import librosa
import webrtcvad
import numpy as np
import matplotlib.pyplot as plt
from typing import List

class SpeechActivityDetector:
    """
    A class for speech activity detection using WebRTC VAD.
    """
    def __init__(self, audio_path: str, sample_rate: int = 16000, vad_mode: int = 3, chunk_duration: float = 0.03, repetitions: int = 11):
        """
        Initialize the SpeechActivityDetector.

        Args:
            audio_path (str): Path to the input audio file.
            sample_rate (int, optional): Sample rate of the audio. Default is 16000.
            vad_mode (int, optional): VAD mode (0-3). Default is 3.
            chunk_duration (float, optional): Duration of audio chunks (in seconds). Default is 0.03.
            repetitions (int, optional): Number of repetitions for predictions. Default is 11.
        """
        self.audio_path = audio_path
        self.sample_rate = sample_rate
        self.vad_mode = vad_mode
        self.chunk_duration = chunk_duration
        self.repetitions = repetitions

    def load_audio(self):
        """
        Load the audio data from the specified file.
        """
        self.audio_data, self.sample_rate = librosa.load(self.audio_path, sr=self.sample_rate)

    def preprocess_audio(self):
        """
        Preprocess the loaded audio data.
        """
        self.samples = (32767 * self.audio_data).astype(np.int16)
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.num_chunks = len(self.samples) // self.chunk_size

    def detect_speech_activity(self):
        """
        Detect speech activity in the audio using WebRTC VAD.
        """
        vad = webrtcvad.Vad()
        vad.set_mode(self.vad_mode)
        self.predictions = [1 if vad.is_speech(self.samples[i * self.chunk_size:(i + 1) * self.chunk_size].tobytes(), self.sample_rate) else 0 for i in range(self.num_chunks)]

    def plot_results(self):
        """
        Plot the audio wave and speech activity predictions.
        """
        time = librosa.times_like(self.audio_data)
        self.predictions = [item*0.55 for item in self.predictions for _ in range(self.repetitions)]

        plt.figure(figsize=(6, 3))
        plt.plot(time, self.audio_data, linewidth=0.5, label='Audio Wave')
        plt.plot(self.predictions, linewidth=0.75, color='r', label='Predictions')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Speech Activity Detection')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Speech Activity Detection using WebRTC VAD")
    parser.add_argument("--audio_path", default='audio/test.wav',  help="Path to the input audio file")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate of the audio")
    parser.add_argument("--vad_mode", type=int, default=3, help="VAD mode (0-3)")
    parser.add_argument("--chunk_duration", type=float, default=0.03, help="Duration of audio chunks (in seconds)")
    parser.add_argument("--repetitions", type=int, default=11, help="Number of repetitions for predictions")

    args = parser.parse_args()

    detector = SpeechActivityDetector(args.audio_path, args.sample_rate, args.vad_mode, args.chunk_duration, args.repetitions)
    detector.load_audio()
    detector.preprocess_audio()
    detector.detect_speech_activity()
    detector.plot_results()

if __name__ == '__main__':
    main()
