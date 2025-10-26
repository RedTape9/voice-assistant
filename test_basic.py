"""
Simple test to verify the assistant's core functionality
This test doesn't require microphone or speech recognition
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Testing Voice Assistant Components...")
print("=" * 50)

# Test 1: Configuration
print("\n1. Testing configuration import...")
try:
    from config import WAKE_WORD, MODEL_NAME, VOICE_RATE
    print(f"   ✓ Wake word: {WAKE_WORD}")
    print(f"   ✓ Model: {MODEL_NAME}")
    print(f"   ✓ Voice rate: {VOICE_RATE}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Voice Output (without actually speaking)
print("\n2. Testing voice output initialization...")
try:
    from voice_output import VoiceOutput
    # Note: We won't actually speak to avoid audio output during testing
    print("   ✓ VoiceOutput class imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Assistant tools (without API key)
print("\n3. Testing assistant tools...")
try:
    from assistant import get_current_time, get_current_date, calculate
    
    time_result = get_current_time()
    print(f"   ✓ Current time: {time_result}")
    
    date_result = get_current_date()
    print(f"   ✓ Current date: {date_result}")
    
    calc_result = calculate("2 + 2")
    print(f"   ✓ Calculator (2+2): {calc_result}")
    
    calc_result2 = calculate("10 * 5")
    print(f"   ✓ Calculator (10*5): {calc_result2}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Assistant initialization (only if API key is available)
print("\n4. Testing assistant initialization...")
api_key = os.getenv("OPENAI_API_KEY")
if api_key and api_key != "your_openai_api_key_here":
    try:
        from assistant import Assistant
        assistant = Assistant(api_key)
        print("   ✓ Assistant initialized successfully")
        print("   ✓ Tools available:", [tool.name for tool in assistant.tools])
    except Exception as e:
        print(f"   ✗ Error: {e}")
else:
    print("   ⚠ Skipped (no valid API key in .env)")

print("\n" + "=" * 50)
print("Basic component tests completed!")
print("\nNote: Full voice testing requires:")
print("  - Microphone hardware")
print("  - Valid OpenAI API key in .env file")
print("  - Running main.py")
