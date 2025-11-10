# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Alloy Voice Assistant** is a multimodal voice assistant combining local speech-to-text (Whisper), webcam vision, LLM agent capabilities (GPT-4o/Ollama), and text-to-speech (Piper TTS). Supports Russian and German with runtime language switching via hotkeys.

## Development Setup

### Installation
```bash
# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Git Bash (Windows)
source .venv/bin/activate      # Linux/macOS

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# If pyaudio fails on Windows:
pip install pipwin
pipwin install pyaudio

# On Linux:
sudo apt install portaudio19-dev python3-pyaudio ffmpeg
```

### Configuration
1. Copy `.env.example` to `.env`
2. Add your OpenAI API key (or configure Ollama for local use)
3. Download Piper TTS models (see README.md section 4)

### Running the Application
```bash
python assistant.py
```

**Hotkeys during runtime:**
- `1` = Switch to Russian
- `2` = Switch to German
- `q` or `ESC` = Exit

### Finding Hardware Indices
```bash
# Find webcam index:
python -c "import cv2; [print(f'Camera {i}: Available') for i in range(10) if cv2.VideoCapture(i).isOpened()]"

# Find audio device index (for TTS output):
python list_audio_devices.py
```

## Architecture

### Three-Component Design

**1. WebcamStream (lines 64-131 in assistant.py)**
- Thread-safe webcam access with background frame capture
- Runs in separate daemon thread to prevent blocking
- Provides Base64-encoded JPEG frames for LLM vision API
- Uses lock-based synchronization for thread safety

**2. Assistant (lines 133-296 in assistant.py)**
- Manages LangChain agent executor with OpenAI function calling
- Handles multilingual TTS with lazy model loading (models cached per language)
- Maintains chat history via `ChatMessageHistory`
- Creates new agent when language is switched (via `_update_agent()`)

**3. Main Loop (lines 299-379 in assistant.py)**
- OpenCV window displays webcam feed with language overlay
- Background audio listening via `SpeechRecognition.listen_in_background()`
- Keyboard event handling for language switching
- Coordinates all components (webcam, STT, agent, TTS)

### Threading Model
- **Main thread**: OpenCV window + keyboard input
- **Webcam thread**: Background frame capture (`WebcamStream._update()`)
- **Audio thread**: Background microphone listening (managed by SpeechRecognition)
- **Synchronous**: Agent execution and TTS playback (blocks until complete)

### Language Configuration System
Language settings are centralized in `LANGUAGE_CONFIG` dictionary (lines 32-45 in assistant.py):
```python
LANGUAGE_CONFIG = {
    "ru": {
        "whisper_lang": "ru",           # STT language
        "response_lang": "Russian",      # LLM response language
        "piper_model": "path/to/model",  # TTS model path
        "display_name": "Russisch"       # UI display name
    }
}
```

When language is switched:
1. Global `CURRENT_LANGUAGE` variable is updated
2. Agent is recreated with new system prompt (line 161)
3. Next STT recognition uses new Whisper language (line 317)
4. Next TTS uses corresponding Piper model (line 210)

### LangChain Agent Setup
- **Agent type**: OpenAI function calling agent (`create_openai_functions_agent`)
- **Tools**: Web search (DuckDuckGo), weather (wttr.in), time, calculator
- **Vision**: Webcam image injected as Base64 in every prompt (line 283)
- **Context**: System prompt includes current date/time to prevent outdated responses (line 264)
- **History**: `ChatMessageHistory` maintains conversation context (line 140)

### Tools Module (tools.py)
Four LangChain tools exported as `DEFAULT_TOOLS`:
- `web_search`: DuckDuckGo search via `ddgs` package
- `weather`: wttr.in API (currently hardcoded to Russian format on line 23)
- `current_time`: Returns formatted date/time (currently hardcoded to Russian on line 35)
- `calculator`: Evaluates math expressions using restricted `eval()` (line 45)

## Known Issues and Limitations

### Critical Issues to Address
1. **Hardcoded audio device** (line 252 in assistant.py): Ignores `.env` configuration, hardcoded to index 3
2. **Unsafe calculator eval** (line 45 in tools.py): While restricted, still vulnerable to DoS via deeply nested expressions
3. **Resource leak in TTS** (lines 246-257 in assistant.py): PyAudio not cleaned up if exception occurs during playback
4. **Race condition** (line 150 in assistant.py): Global `CURRENT_LANGUAGE` accessed without synchronization from multiple threads

### Design Limitations
1. **Blocking TTS**: Speech synthesis blocks all other operations (no interrupt capability)
2. **Unbounded chat history**: Memory grows indefinitely in long sessions
3. **Agent recreation on language switch**: Inefficient, loses internal state
4. **Full frame copying**: Every webcam read creates a full copy (line 118)

### Language-Specific Hardcoding
- Weather tool always returns Russian format (line 23 in tools.py)
- Time tool always returns Russian text (line 35 in tools.py)
- These should respect `CURRENT_LANGUAGE` global variable

## Adding New Features

### Adding a New Language
1. Download Piper model from https://huggingface.co/rhasspy/piper-voices
2. Add entry to `LANGUAGE_CONFIG` dictionary in assistant.py
3. Add environment variable to `.env.example`
4. Add hotkey handler in main loop (around line 367)

Example:
```python
"en": {
    "whisper_lang": "en",
    "response_lang": "English",
    "piper_model": os.getenv("PIPER_MODEL_EN"),
    "display_name": "English"
}
```

### Adding a New Tool
In `tools.py`:
```python
def your_tool(input: str) -> str:
    """Tool description for the LLM."""
    # Implementation
    return result

DEFAULT_TOOLS.append(Tool(
    name="your_tool",
    func=your_tool,
    description="Description for the agent. Input: what format expected."
))
```

### Adding Dependencies
1. Add to `requirements.txt` with version constraint
2. Document any platform-specific installation steps in README.md
3. For optional dependencies, use `; platform_system=="..."` suffix

## Security Considerations

- `.env` file contains API keys and must never be committed (already in `.gitignore`)
- Calculator tool uses `eval()` - only safe for trusted inputs
- No input sanitization before sending to LLM
- No rate limiting on API calls or tool usage
- Webcam images sent to external API without user consent per-image

## Environment Variables Reference

Essential variables (must be set):
- `OPENAI_API_KEY`: OpenAI API key or "dummy" for Ollama
- `OPENAI_BASE_URL`: API endpoint URL
- `OPENAI_MODEL`: Model name (must support vision)
- `PIPER_MODEL_RU`: Path to Russian TTS model
- `PIPER_MODEL_DE`: Path to German TTS model

Optional variables (defined in .env.example but not fully implemented):
- `AUDIO_OUTPUT_DEVICE`: Currently ignored (hardcoded to 3 on line 252)
- `AGENT_MAX_ITERATIONS`: Defined but unused
- `WEATHER_LANG`: Defined but unused (hardcoded to Russian)
- `WEB_SEARCH_MAX_RESULTS`: Defined but unused (hardcoded to 3)

## Testing and Debugging

### Testing Hardware
```bash
# Test webcam
python -c "import cv2; cap=cv2.VideoCapture(0); print('OK' if cap.read()[0] else 'FAILED')"

# Test audio devices
python list_audio_devices.py
```

### Debugging Agent Execution
- Set `verbose=True` in `AgentExecutor` (line 293) to see tool calls and reasoning
- Agent is configured with `handle_parsing_errors=True` to prevent crashes on malformed tool calls

### Common Error Patterns
- **"Kamera nicht lesbar"**: Wrong `CAMERA_INDEX` or camera in use by another application
- **PyAudio errors**: Wrong `AUDIO_OUTPUT_DEVICE` or device not available
- **"Kein Piper-Model"**: TTS model files not downloaded or wrong path in `.env`
- **Vision not working**: Model doesn't support vision (e.g., gpt-3.5-turbo) or webcam failing

## Platform-Specific Notes

### Windows
- Uses DirectShow backend for webcam (CAP_DSHOW) with fallback to default (line 68)
- Requires `onnxruntime-gpu` for GPU-accelerated TTS (CUDA 11.8+)
- PyAudio installation often fails - use `pipwin` workaround

### Linux
- Requires system packages: `portaudio19-dev`, `python3-pyaudio`, `ffmpeg`
- No special webcam backend needed
- Use CPU version of onnxruntime unless CUDA available
