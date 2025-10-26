# Voice Assistant

Your personal voice assistant with LangChain tools - interact with AI using your voice!

## Features

- 🎤 **Voice Input**: Speak naturally to your assistant using speech recognition
- 🔊 **Voice Output**: Hear responses through text-to-speech
- 🤖 **AI-Powered**: Uses OpenAI's GPT models through LangChain
- 🛠️ **Multiple Tools**:
  - Wikipedia search for factual information
  - Web search for current information
  - Calculator for mathematical operations
  - Current time and date

## Prerequisites

- Python 3.8 or higher
- Microphone for voice input
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RedTape9/voice-assistant.git
cd voice-assistant
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file:
```bash
cp .env.example .env
# Edit .env and add your API key
```

## Usage

Run the voice assistant:
```bash
python main.py
```

### How to interact:

1. Wait for the assistant to initialize
2. Say the wake word "**assistant**" followed by your question or command
3. Examples:
   - "Assistant, what's the weather like today?"
   - "Assistant, what is 25 times 4?"
   - "Assistant, tell me about Albert Einstein"
   - "Assistant, what time is it?"
   - "Assistant, search for the latest news on AI"

4. To exit, say "exit", "quit", or "goodbye"

## Configuration

You can customize the assistant by editing `config.py`:

- `VOICE_RATE`: Speed of speech (default: 150)
- `VOICE_VOLUME`: Volume level 0.0 to 1.0 (default: 0.9)
- `WAKE_WORD`: Wake word to activate the assistant (default: "assistant")
- `MODEL_NAME`: OpenAI model to use (default: "gpt-3.5-turbo")
- `MODEL_TEMPERATURE`: Response creativity (default: 0.7)

## Project Structure

```
voice-assistant/
├── main.py              # Main application entry point
├── assistant.py         # LangChain agent with tools
├── voice_input.py       # Speech recognition handler
├── voice_output.py      # Text-to-speech handler
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment variables
└── README.md           # This file
```

## Available Tools

The assistant has access to the following tools:

1. **Wikipedia**: Look up factual information
2. **Search**: Search the internet for current information
3. **Calculator**: Perform mathematical calculations
4. **CurrentTime**: Get the current time
5. **CurrentDate**: Get the current date

## Troubleshooting

### Microphone issues
- Make sure your microphone is properly connected
- Check system audio input settings
- The assistant calibrates for ambient noise on startup

### API errors
- Verify your OpenAI API key is correct in the `.env` file
- Check your OpenAI account has available credits

### Speech recognition issues
- Speak clearly and at a moderate pace
- Reduce background noise
- Adjust `RECOGNITION_TIMEOUT` in `config.py` if needed

## License

MIT License - feel free to use and modify as needed.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
