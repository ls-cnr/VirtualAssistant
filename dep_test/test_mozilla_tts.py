from TTS.api import TTS
import pygame
import soundfile as sf
import numpy as np
import tempfile
import os

def test_mozilla_tts(text, model_name):
    print(f"\nTesting Mozilla TTS with model: {model_name}")
    print(f"Text: '{text}'")

    # Initialize pygame mixer
    pygame.mixer.init()

    try:
        # Initialize TTS with selected model
        print("Loading model...")
        tts = TTS(model_name=model_name, progress_bar=True)

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            print("Generating speech...")
            # Generate audio file
            tts.tts_to_file(text=text, file_path=temp_file.name)

            print("Playing speech...")
            # Play the audio
            pygame.mixer.music.load(temp_file.name)
            pygame.mixer.music.play()

            # Wait for audio to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            # Cleanup
            pygame.mixer.music.unload()

        # Remove temporary file
        os.unlink(temp_file.name)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        pygame.mixer.quit()

if __name__ == "__main__":
    # List available models
    print("Available models:")
    all_models = TTS().list_models()
    for idx, model in enumerate(all_models):
        print(f"{idx}: {model}")

    # Test English with different models
    english_text = "Hello, this is a test of the Mozilla Text to Speech system. How does it sound?"
    test_mozilla_tts(english_text, "tts_models/en/ljspeech/tacotron2-DDC")
    test_mozilla_tts(english_text, "tts_models/en/vctk/vits")

    # Test Italian with multi-lingual model
    italian_text = "Ciao, questo è un test del sistema di sintesi vocale Mozilla. Come ti sembra?"
    test_mozilla_tts(italian_text, "tts_models/multilingual/multi-dataset/xtts_v2")

    # Test a longer phrase
    long_text = ("This is a longer phrase to test how the system handles multiple sentences. " +
                "We want to see if there are any issues with timing, pronunciation, or natural pauses. " +
                "Does it sound natural? How is the intonation?")
    test_mozilla_tts(long_text, "tts_models/en/vctk/vits")
