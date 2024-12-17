# providers/base.py
from abc import ABC, abstractmethod
import numpy as np

class AudioProvider(ABC):
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Get the sample rate"""
        pass

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Get the chunk size"""
        pass

    @abstractmethod
    def start_stream(self) -> None:
        """Start the audio stream"""
        pass

    @abstractmethod
    def read_chunk(self) -> np.ndarray:
        """Read a chunk of audio data"""
        pass

    @abstractmethod
    def stop_stream(self) -> None:
        """Stop the audio stream"""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources"""
        pass

class VADProvider(ABC):
    @abstractmethod
    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int) -> bool:
        """Determine if audio chunk contains speech"""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources"""
        pass

class TranscriptionProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_data: np.ndarray) -> str:
        """Convert audio to text"""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources"""
        pass

class LLMProvider(ABC):
    @abstractmethod
    def get_response(self, text: str, system_prompt: str) -> str:
        """Get response from LLM"""
        pass

class TextFilterProvider(ABC):
    @abstractmethod
    def filter(self, text: str) -> str:
        """Filter text to remove unwanted elements"""
        pass

class TTSProvider(ABC):
    @abstractmethod
    def speak(self, text: str, language: str) -> None:
        """Convert text to speech"""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources"""
        pass
