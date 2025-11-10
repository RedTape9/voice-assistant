"""
Voice output handler using text-to-speech
"""
from typing import Any
import pyttsx3
from config import VOICE_RATE, VOICE_VOLUME


class VoiceOutput:
    """Handles voice output using text-to-speech"""

    def __init__(self) -> None:
        """Initialize the text-to-speech engine with configured settings"""
        self.engine: Any = pyttsx3.init()
        self.engine.setProperty('rate', VOICE_RATE)
        self.engine.setProperty('volume', VOICE_VOLUME)

    def speak(self, text: str) -> None:
        """
        Convert text to speech and play it

        Args:
            text (str): The text to speak
        """
        print(f"Assistant: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def __del__(self) -> None:
        """Cleanup: Stop the TTS engine when object is destroyed"""
        if hasattr(self, 'engine') and self.engine is not None:
            try:
                self.engine.stop()
            except Exception:
                # Ignore errors during cleanup
                pass
