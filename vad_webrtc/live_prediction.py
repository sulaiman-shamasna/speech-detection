import webrtcvad
import numpy as np
import sounddevice as sd
import argparse
from typing import Any, Union

class SpeechDetector:
    def __init__(self, sample_rate: int, vad_mode: int):
        """
        Initialize the VoiceActivityDetector.

        Args:
            sample_rate (int): Audio sample rate.
            vad_mode (int): VAD mode (0, 1, 2, or 3).

        Returns:
            None
        """
        self.sample_rate = sample_rate
        self.vad_mode = vad_mode
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(vad_mode)

    def process_audio(self, indata: np.ndarray, frames: int, time: Any, status: Union[int, None]) -> None:
        """
        Callback function to process audio data and detect voice activity using WebRTC VAD.

        Args:
            indata (np.ndarray): Input audio data.
            frames (int): Number of frames.
            time (Any): Timestamp (not used in this example).
            status (Union[int, None]): Status of the audio stream.

        Returns:
            None
        """
        if status:
            print(status)

        samples = (32767 * indata).astype(np.int16)

        detected_voice = self.vad.is_speech(samples.tobytes(), self.sample_rate)

        if detected_voice:
            print("Voice Activity Detected")
        else:
            print('zZ')

    def run(self):
        """
        Start speech detection.

        Returns:
            None
        """
        try:
            with sd.InputStream(
                channels=1,
                callback=self.process_audio,
                samplerate=self.sample_rate,
                blocksize=int(0.03 * self.sample_rate)
            ):
                while True:
                    pass
        except KeyboardInterrupt:
            print("Interrupted by the user")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Voice Activity Detection using WebRTC VAD")
    parser.add_argument("--sample-rate", type=int, default=8000, help="Audio sample rate")
    parser.add_argument("--vad-mode", type=int, default=3, choices=[0, 1, 2, 3], help="VAD mode (0, 1, 2, or 3)")

    args = parser.parse_args()
    vad_detector = SpeechDetector(sample_rate=args.sample_rate, vad_mode=args.vad_mode)
    vad_detector.run()

if __name__ == "__main__":
    main()