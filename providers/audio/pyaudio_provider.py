import pyaudio
import numpy as np
from typing import Optional
from ..base import AudioProvider

class PyAudioProvider(AudioProvider):
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 512):
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        self.format = pyaudio.paFloat32
        self.channels = 1

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def start_stream(self) -> None:
        """Start the audio stream"""
        if self.stream is None:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            self.stream.start_stream()

    def read_chunk(self) -> np.ndarray:
        """Read a chunk of audio data"""
        if self.stream is None:
            raise RuntimeError("Stream not started")

        try:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            return np.frombuffer(data, dtype=np.float32)
        except Exception as e:
            print(f"Error reading audio: {e}")
            return np.zeros(self.chunk_size, dtype=np.float32)

    def stop_stream(self) -> None:
        """Stop the audio stream"""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def cleanup(self) -> None:
        """Cleanup resources"""
        self.stop_stream()
        self.audio.terminate()
