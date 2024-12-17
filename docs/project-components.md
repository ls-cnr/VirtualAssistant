# Voice Assistant Documentation

## Core Components

### VoiceAssistant (core/assistant.py)
The main class that orchestrates all the components of the voice assistant system.

**Key Features:**
- Handles audio input processing
- Manages speech detection
- Coordinates transcription, LLM responses, and text-to-speech output
- Supports multiple languages

**Main Methods:**
- `__init__`: Initializes all providers and sets up logging
- `process_audio_chunk`: Processes single audio chunks for speech detection
- `process_recording`: Handles complete audio recordings for transcription
- `get_response`: Gets responses from the LLM
- `speak_response`: Converts text to speech
- `run`: Main loop of the assistant

## Base Providers (providers/base.py)

The project uses abstract base classes to define interfaces that each provider must implement:

### AudioProvider
Base class for audio input/output operations.

**Abstract Methods:**
- `sample_rate`: Property that returns the audio sample rate
- `chunk_size`: Property that returns the size of audio chunks
- `start_stream`: Initializes and starts the audio stream
- `read_chunk`: Reads a chunk of audio data
- `stop_stream`: Stops the audio stream
- `cleanup`: Releases resources

### VADProvider
Base class for Voice Activity Detection.

**Abstract Methods:**
- `is_speech(audio_chunk, sample_rate)`: Determines if an audio chunk contains speech
- `cleanup`: Releases resources

### TranscriptionProvider
Base class for Speech-to-Text operations.

**Abstract Methods:**
- `transcribe(audio_data)`: Converts audio data to text
- `cleanup`: Releases resources

### LLMProvider
Base class for Large Language Model interactions.

**Abstract Methods:**
- `get_response(text, system_prompt)`: Gets response from the LLM

### TextFilterProvider
Base class for text filtering operations.

**Abstract Methods:**
- `filter(text)`: Filters text to remove unwanted elements

### TTSProvider
Base class for Text-to-Speech operations.

**Abstract Methods:**
- `speak(text, language)`: Converts text to speech
- `cleanup`: Releases resources

## Provider Implementations

### PyAudioProvider (providers/audio/pyaudio_provider.py)
Handles audio input/output operations using PyAudio.

**Key Features:**
- Manages audio stream initialization and cleanup
- Configurable sample rate and chunk size
- Error handling for audio operations

**Main Methods:**
- `start_stream`: Initializes and starts the audio input stream
- `read_chunk`: Reads a chunk of audio data
- `stop_stream`: Stops and closes the audio stream
- `cleanup`: Releases audio resources

### SileroVAD (providers/vad/silero_provider.py)
Voice Activity Detection using the Silero VAD model.

**Key Features:**
- Uses PyTorch for inference
- CPU-based processing
- Efficient speech detection

**Main Methods:**
- `is_speech`: Determines if an audio chunk contains speech
- `cleanup`: Releases resources

### GoogleTTS (providers/tts/google_provider.py)
Text-to-Speech provider using Google's gTTS service.

**Key Features:**
- Multi-language support
- Audio playback using pygame
- Temporary file handling

**Main Methods:**
- `speak`: Converts text to speech and plays it
- `cleanup`: Cleans up pygame resources

### WhisperProvider (providers/transcription/whisper_provider.py)
Speech-to-Text provider using OpenAI's Whisper model.

**Key Features:**
- Uses Hugging Face's transformers
- Supports multiple languages
- Optimized for Apple Silicon (MPS)

**Main Methods:**
- `transcribe`: Converts audio to text
- `cleanup`: Releases resources

### SpeechFilter (providers/filter/speech_filter.py)
Text filtering provider that removes non-speakable elements from text.

**Key Features:**
- Removes Unicode emojis
- Removes text-based emoticons
- Removes text between asterisks (actions/emotions)
- Removes text-based emoji codes
- Cleans up multiple spaces

**Methods:**
- `filter`: Removes non-speakable elements from text using regex patterns
- `print_filtered`: Debug utility to compare original and filtered text

### OllamaLLM (providers/llm/ollama_provider.py)
Large Language Model provider using Ollama.

**Key Features:**
- Conversation memory management
- System prompt customization
- Local LLM processing

**Main Methods:**
- `get_response`: Gets LLM response for input text
- `cleanup`: Clears conversation memory

## Configuration

### LanguageConfig (config/language_config.py)
Configuration for multi-language support.

**Key Features:**
- Language-specific error messages
- Custom LLM system prompts per language
- Whisper language configuration

**Available Languages:**
- English (en)
- Italian (it)

## Architecture Notes
- Modular design with clear separation of concerns
- Each provider implements a specific interface defined in base.py
- Error handling at each level
- Configurable components
- Support for multiple languages

## Dependencies
Main project dependencies include:
- PyAudio for audio handling
- PyTorch for ML models
- Transformers for Whisper
- Langchain for LLM integration
- gTTS for text-to-speech
- Pygame for audio playback