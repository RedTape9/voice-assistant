"""
Voice output handler using text-to-speech
"""
import pyttsx3
from config import VOICE_RATE, VOICE_VOLUME


class VoiceOutput:
    """Handles voice output using text-to-speech"""
    
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', VOICE_RATE)
        self.engine.setProperty('volume', VOICE_VOLUME)
    
    def speak(self, text):
        """
        Convert text to speech and play it
        
        Args:
            text (str): The text to speak
        """
        print(f"Assistant: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
