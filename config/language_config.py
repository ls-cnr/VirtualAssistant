# config/language_config.py
from dataclasses import dataclass
from typing import Dict

@dataclass
class LanguageConfig:
    code: str
    whisper_language: str
    error_messages: Dict[str, str]
    llm_system_prompt: str

LANGUAGE_CONFIGS = {
    "en": LanguageConfig(
        code="en",
        whisper_language="en",
        error_messages={
            "not_understood": "I didn't understand. Could you repeat that?",
            "processing_error": "Sorry, I encountered an error processing your request."
        },
        llm_system_prompt="You are a friendly voice assistant. Keep responses concise and natural."
    ),
    "it": LanguageConfig(
        code="it",
        whisper_language="it",
        error_messages={
            "not_understood": "Non ho capito. Potresti ripetere?",
            "processing_error": "Mi dispiace, ho avuto un problema nell'elaborare la richiesta."
        },
        llm_system_prompt="Sei un assistente vocale amichevole che parla in italiano. Mantieni le risposte concise e naturali."
    )
}
