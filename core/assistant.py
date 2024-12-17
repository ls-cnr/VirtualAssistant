from typing import Optional
import numpy as np
from time import sleep
from providers.base import (
    AudioProvider,
    VADProvider,
    LLMProvider,
    TTSProvider,
    TextFilterProvider,
    TranscriptionProvider
)
from config.language_config import LANGUAGE_CONFIGS, LanguageConfig
import logging
from enum import Enum

class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

class VoiceAssistant:
    def __init__(self,
                 audio_provider: AudioProvider,
                 vad_provider: VADProvider,
                 transcription_provider: TranscriptionProvider,
                 llm_provider: LLMProvider,
                 text_filter_provider: TextFilterProvider,
                 tts_provider: TTSProvider,
                 language: str = "en",
                 log_level: LogLevel = LogLevel.INFO):

        # Setup logging
        logging.basicConfig(
            format='%(levelname)s: %(message)s',
            level=log_level.value
        )
        self.logger = logging.getLogger(__name__)

        self.audio = audio_provider
        self.vad = vad_provider
        self.transcriber = transcription_provider
        self.llm = llm_provider
        self.text_filter = text_filter_provider
        self.tts = tts_provider

        # Set language configuration
        self.lang_config = self._get_language_config(language)

        # State
        self.is_running = False

    def _get_language_config(self, language: str) -> LanguageConfig:
        if language not in LANGUAGE_CONFIGS:
            print(f"Language {language} not supported, falling back to English")
            language = "en"
        return LANGUAGE_CONFIGS[language]

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> bool:
        """Process single audio chunk and return True if speech was detected"""
        try:
            if self.vad.is_speech(audio_chunk, self.audio.sample_rate):
                self.logger.debug("Speech detected!")
                return True
        except Exception as e:
            self.logger.error(f"Error processing audio chunk: {e}")
        return False

    def process_recording(self, audio_frames: list[np.ndarray]) -> Optional[str]:
        """Process complete recording and return transcription"""
        try:
            if not audio_frames:
                return None
            # Concatena tutti i frame audio
            full_audio = np.concatenate(audio_frames)
            # Trascrivi l'audio
            return self.transcriber.transcribe(full_audio)
        except Exception as e:
            print(f"Error processing recording: {e}")
            return None

    def get_response(self, text: str) -> str:
        """Get response from LLM"""
        try:
            return self.llm.get_response(text, self.lang_config.llm_system_prompt)
        except Exception as e:
            print(f"Error getting response: {e}")
            return self.lang_config.error_messages["processing_error"]

    def speak_response(self, text: str) -> None:
        """Speak the response"""
        try:
            filtered_text = self.text_filter.filter(text)
            self.tts.speak(filtered_text, self.lang_config.code)

        except Exception as e:
            print(f"Error speaking response: {e}")

    def run(self) -> None:
        """Main loop"""
        self.logger.info("Starting Voice Assistant. Press Ctrl+C to exit.")

        self.is_running = True
        self.audio.start_stream()

        try:
            frames = []
            is_recording = False
            silence_counter = 0

            while self.is_running:
                # Read audio chunk
                audio_chunk = self.audio.read_chunk()

                # Check for speech
                if self.process_audio_chunk(audio_chunk):
                    is_recording = True
                    silence_counter = 0
                    frames.append(audio_chunk)
                elif is_recording:
                    silence_counter += 1
                    frames.append(audio_chunk)

                    # Stop recording after ~1 second of silence
                    if silence_counter > int(self.audio.sample_rate / self.audio.chunk_size):
                        print("Processing speech...")
                        # Transcribe the recorded audio
                        text = self.process_recording(frames)
                        if text:
                            print(f"You said: {text}")

                            response = self.get_response(text)
                            print(f"Assistant: {response}")

                            self.speak_response(response)

                        # Reset for next interaction
                        frames = []
                        is_recording = False
                        silence_counter = 0

                # Small sleep to prevent CPU overload
                sleep(0.001)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup all providers"""
        self.is_running = False
        self.audio.cleanup()
        self.vad.cleanup()
        self.transcriber.cleanup()
        self.tts.cleanup()
