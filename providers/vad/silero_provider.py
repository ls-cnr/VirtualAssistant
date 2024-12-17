import torch
import numpy as np
from typing import Union, Any, cast
from torch import nn
from ..base import VADProvider

class SileroVAD(VADProvider):
    def __init__(self):
        print("Initializing Silero VAD...")
        # Usiamo Any per il valore restituito da torch.hub.load e poi facciamo il cast
        vad_model: Any = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=True,
            trust_repo=True
        )

        self.model: nn.Module
        # Verifichiamo il tipo e facciamo il cast appropriato
        if isinstance(vad_model, tuple):
            self.model = cast(nn.Module, vad_model[0])
        else:
            self.model = cast(nn.Module, vad_model)

        self.model = self.model.to("cpu")
        self.model.eval()

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int) -> bool:
        with torch.no_grad():
            # Crea una copia writeable dell'array
            audio_data = np.array(audio_chunk, dtype=np.float32, copy=True)
            tensor = torch.FloatTensor(audio_data)
            return self.model(tensor, sample_rate).item() > 0.5

    def cleanup(self) -> None:
        pass  # Nothing to cleanup for Silero VAD
