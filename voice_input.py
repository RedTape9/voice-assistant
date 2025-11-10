"""
Voice input handler using speech recognition
"""
from typing import Optional
import speech_recognition as sr
from config import RECOGNITION_TIMEOUT, RECOGNITION_PHRASE_TIME_LIMIT


class VoiceInput:
    """Handles voice input using speech recognition"""

    def __init__(self) -> None:
        """
        Initialize voice input with microphone availability check

        Raises:
            RuntimeError: If no microphone is available
        """
        self.recognizer: sr.Recognizer = sr.Recognizer()

        # Check if microphone is available
        try:
            self.microphone: sr.Microphone = sr.Microphone()
        except OSError as e:
            raise RuntimeError(
                "No microphone found. Please connect a microphone and try again."
            ) from e

        # Adjust for ambient noise
        try:
            with self.microphone as source:
                print("Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            raise RuntimeError(
                f"Failed to access microphone. It may be in use by another application: {e}"
            ) from e
    
    def listen(self) -> Optional[str]:
        """
        Listen for voice input and convert to text

        Returns:
            Optional[str]: The recognized text, or None if recognition failed
        """
        with self.microphone as source:
            print("Listening...")
            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=RECOGNITION_TIMEOUT,
                    phrase_time_limit=RECOGNITION_PHRASE_TIME_LIMIT
                )

                print("Processing speech...")
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text

            except sr.WaitTimeoutError:
                print("No speech detected")
                return None
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return None
