"""
Voice input handler using speech recognition
"""
import speech_recognition as sr
from config import RECOGNITION_TIMEOUT, RECOGNITION_PHRASE_TIME_LIMIT


class VoiceInput:
    """Handles voice input using speech recognition"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        with self.microphone as source:
            print("Calibrating microphone for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
    
    def listen(self):
        """
        Listen for voice input and convert to text
        
        Returns:
            str: The recognized text, or None if recognition failed
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
