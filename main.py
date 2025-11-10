"""
Main voice assistant application
"""
import os
from typing import Optional
from dotenv import load_dotenv
from voice_input import VoiceInput
from voice_output import VoiceOutput
from assistant import Assistant
from config import WAKE_WORD


def main() -> None:
    """Main function to run the voice assistant"""
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        print("See .env.example for the format.")
        return
    
    print("=" * 50)
    print("Voice Assistant with LangChain Tools")
    print("=" * 50)
    print()
    
    # Initialize components
    try:
        print("Initializing voice assistant...")
        voice_input = VoiceInput()
        voice_output = VoiceOutput()
        assistant = Assistant(api_key)
        print("Voice assistant initialized successfully!")
        print()
    except Exception as e:
        print(f"Error initializing assistant: {e}")
        return
    
    # Welcome message
    welcome_msg = f"Hello! I'm your voice assistant. Say '{WAKE_WORD}' followed by your question or command."
    voice_output.speak(welcome_msg)
    print()
    print("Commands:")
    print(f"  - Say '{WAKE_WORD}' followed by your question")
    print("  - Say 'exit' or 'quit' to close the assistant")
    print()
    
    # Main loop
    while True:
        try:
            # Listen for input
            user_input = voice_input.listen()
            
            if user_input is None:
                continue
            
            # Convert to lowercase for easier processing
            user_input_lower = user_input.lower()
            
            # Check for exit commands
            if any(word in user_input_lower for word in ["exit", "quit", "goodbye", "stop"]):
                farewell_msg = "Goodbye! Have a great day!"
                voice_output.speak(farewell_msg)
                break
            
            # Check for wake word
            if WAKE_WORD.lower() in user_input_lower:
                # Remove wake word from input
                query = user_input_lower.replace(WAKE_WORD.lower(), "").strip()
                
                if not query:
                    voice_output.speak("Yes? How can I help you?")
                    continue
                
                # Process the query
                print(f"\nProcessing: {query}")
                response = assistant.process(query)
                
                # Speak the response
                voice_output.speak(response)
                print()
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            voice_output.speak("Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            voice_output.speak("I encountered an error. Please try again.")


if __name__ == "__main__":
    main()
