# providers/tts/google_provider.py
from gtts import gTTS
import pygame
import tempfile
import os
from ..base import TTSProvider

class GoogleTTS(TTSProvider):
    def __init__(self):
        pygame.mixer.init()

    def speak(self, text: str, language: str) -> None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                tts = gTTS(text=text, lang=language)
                tts.save(temp_file.name)
                
                pygame.mixer.music.load(temp_file.name)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                    
                pygame.mixer.music.unload()
            os.unlink(temp_file.name)
            
        except Exception as e:
            print(f"Error in TTS: {e}")
            raise

    def cleanup(self) -> None:
        pygame.mixer.quit()
