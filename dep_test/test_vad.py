import torch
import pyaudio
import numpy as np
from typing import Any, Union, cast
import time

class AudioManager:
    def __init__(self, chunk_size: int, sample_rate: int):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.audio = pyaudio.PyAudio()

    def list_input_devices(self) -> None:
        """List all available input devices"""
        print("\nAvailable input devices:")
        for i in range(self.audio.get_device_count()):
            dev_info = self.audio.get_device_info_by_index(i)
            max_channels = cast(Union[int, float], dev_info.get('maxInputChannels', 0))
            if max_channels > 0:
                print(f"Index {i}: {dev_info.get('name')}")

    def create_stream(self) -> pyaudio.Stream:
        """Create and return an audio stream"""
        try:
            return self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=None,  # Use default input device
                stream_callback=None,
                start=False  # Don't start immediately
            )
        except Exception as e:
            self.audio.terminate()
            raise Exception(f"Failed to create audio stream: {str(e)}")

    def cleanup(self) -> None:
        """Cleanup audio resources"""
        self.audio.terminate()

def main():
    # Constants
    CHUNK_SIZE = 512
    SAMPLE_RATE = 16000

    print("Loading Silero VAD model...")
    # Download and initialize Silero VAD
    vad_tup = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=True,
        trust_repo=True,
        verbose=True
    )

    # Il modello è il primo elemento della tupla
    model = vad_tup if torch.nn.Module in type(vad_tup).__mro__ else vad_tup[0]
    print(f"Model loaded successfully. Type: {type(model)}")

    # Use CPU explicitly
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    print(f"\nInitializing audio...")
    audio_manager = AudioManager(CHUNK_SIZE, SAMPLE_RATE)
    audio_manager.list_input_devices()
    stream = audio_manager.create_stream()

    print(f"\nTesting VAD - speak something (running for 5 seconds)...")
    print(f"Using device: {device}")

    try:
        # Start the stream
        stream.start_stream()
        start_time = time.time()

        while time.time() - start_time < 5:  # Run for 5 seconds
            try:
                # Read with timeout to prevent blocking
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)

                # Make data writeable
                audio_data = np.array(audio_data, dtype=np.float32)

                # Convert to tensor and check VAD
                tensor = torch.from_numpy(audio_data).to(device)
                speech_prob = model(tensor, SAMPLE_RATE).item()

                if speech_prob > 0.5:
                    print(f"Speech detected! Probability: {speech_prob:.2f}")

                # Small sleep to prevent CPU overload
                time.sleep(0.001)

            except OSError as e:
                print(f"Warning: {str(e)}")
                continue

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        audio_manager.cleanup()
        print("\nTest completed!")

if __name__ == "__main__":
    main()
