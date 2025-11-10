"""
Configuration settings for the voice assistant
"""

# Voice settings
VOICE_RATE = 150  # Speed of speech
VOICE_VOLUME = 0.9  # Volume level (0.0 to 1.0)

# Recognition settings
RECOGNITION_TIMEOUT = 5  # Seconds to wait for speech
RECOGNITION_PHRASE_TIME_LIMIT = 10  # Max seconds for a single phrase

# Wake word
WAKE_WORD = "assistant"

# Model settings
MODEL_NAME = "gpt-3.5-turbo"
MODEL_TEMPERATURE = 0.7
