import warnings
from transformers import logging

import numpy as np
from transformers import pipeline
from scipy.io import wavfile
import tempfile
from typing import Any
from ..base import TranscriptionProvider

class WhisperProvider(TranscriptionProvider):
    def __init__(self, language: str = "en", device: str = "mps"):
        # Disabilita i warning di transformers
        warnings.filterwarnings("ignore", category=FutureWarning)
        logging.set_verbosity_error()  # Mostra solo errori, non warning

        print("Initializing Whisper...")
        self.stt = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=device,
            generate_kwargs={
                "language": language,
                "task": "transcribe"
            }
        )

    def transcribe(self, audio_data: np.ndarray) -> str:
        try:
            wavfile.write("temp.wav", 16000, audio_data)
            result = self.stt("temp.wav", batch_size=1)
            return result["text"].strip()
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""

    def cleanup(self) -> None:
        pass  # Nothing to cleanup for Whisper
