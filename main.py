from providers.vad.silero_provider import SileroVAD
from providers.llm.ollama_provider import OllamaLLM
from providers.tts.google_provider import GoogleTTS
from providers.audio.pyaudio_provider import PyAudioProvider
from providers.transcription.whisper_provider import WhisperProvider
from providers.filter.speech_filter import SpeechFilter

from core.assistant import VoiceAssistant, LogLevel

def setup_warnings():
    import warnings
    from transformers import logging

    # Disabilita i warning specifici
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Configura logging per transformers
    logging.set_verbosity_error()


def main():
    setup_warnings()

    try:
        # Initialize providers
        audio_provider = PyAudioProvider(sample_rate=16000, chunk_size=512)
        vad_provider = SileroVAD()
        transcription_provider = WhisperProvider(language="en", device="mps")
        llm_provider = OllamaLLM()
        text_filter_provider = SpeechFilter()
        tts_provider = GoogleTTS()

        assistant = VoiceAssistant(
            audio_provider=audio_provider,
            vad_provider=vad_provider,
            transcription_provider=transcription_provider,
            llm_provider=llm_provider,
            text_filter_provider=text_filter_provider,
            tts_provider=tts_provider,
            language="en",
            log_level=LogLevel.INFO  # Mostra solo info e errori
        )

        # Run the assistant
        assistant.run()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
