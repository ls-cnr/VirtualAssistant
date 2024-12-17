from gtts import gTTS
import pygame
import tempfile
import os

def test_tts(text, lang):
    print(f"Testing TTS with text: '{text}' in language: {lang}")

    # Initialize pygame mixer
    pygame.mixer.init()

    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            # Generate audio file
            print("Generating speech...")
            tts = gTTS(text=text, lang=lang)
            tts.save(temp_file.name)

            # Play audio
            print("Playing speech...")
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
    # Test English
    test_tts("Hello, this is a test of the Google Text to Speech system. How does it sound?", "en")

    # Test Italian
    test_tts("Ciao, questo è un test del sistema di sintesi vocale di Google. Come ti sembra?", "it")

    # Test a longer phrase
    test_tts("This is a longer phrase to test how the system handles multiple sentences. " +
             "We want to see if there are any issues with timing, pronunciation, or natural pauses. " +
             "Does it sound natural? How is the intonation?", "en")
