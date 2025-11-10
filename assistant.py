from threading import Lock, Thread
import os
import base64
import cv2
import time
import re
from collections import deque
from datetime import datetime
from typing import Optional
from openai import OpenAI
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

from tools import DEFAULT_TOOLS

# ----- ENV -----
load_dotenv(find_dotenv(), override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

DISABLE_TTS = os.getenv("DISABLE_TTS", "0") == "1"

# Mehrsprachigkeit: Default + verfügbare Sprachen
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "ru")  # 'ru' oder 'de'
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# Audio devices
AUDIO_INPUT_DEVICE = os.getenv("AUDIO_INPUT_DEVICE")  # Mikrofon für STT
AUDIO_INPUT_DEVICE = int(AUDIO_INPUT_DEVICE) if AUDIO_INPUT_DEVICE else None
AUDIO_OUTPUT_DEVICE = os.getenv("AUDIO_OUTPUT_DEVICE")  # Lautsprecher für TTS
AUDIO_OUTPUT_DEVICE = int(AUDIO_OUTPUT_DEVICE) if AUDIO_OUTPUT_DEVICE else None

# Language-Konfiguration (Whisper + Response + TTS)
LANGUAGE_CONFIG = {
    "ru": {
        "whisper_lang": "ru",
        "response_lang": "Russian",
        "piper_model": os.getenv("PIPER_MODEL_RU", "models/piper/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx"),
        "display_name": "Russisch"
    },
    "de": {
        "whisper_lang": "de",
        "response_lang": "German",
        "piper_model": os.getenv("PIPER_MODEL_DE", "models/piper/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx"),
        "display_name": "Deutsch"
    }
}

# Aktuelle Sprache (wird dynamisch umgeschaltet)
CURRENT_LANGUAGE = DEFAULT_LANGUAGE
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY fehlt (.env prüfen)")

# ----- Helper Functions -----
def clean_markdown_for_tts(text: str) -> str:
    """
    Entfernt Markdown-Formatierung für TTS-Ausgabe.

    Konvertiert:
    - **bold** -> bold
    - *italic* -> italic
    - [link](url) -> link
    - # Heading -> Heading
    - etc.
    """
    # Bold und Italic (**, *, __, _)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)

    # Links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Headers (# ## ###)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Code blocks (`)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Strikethrough (~~)
    text = re.sub(r'~~([^~]+)~~', r'\1', text)

    return text

# ----- LLM Setup -----
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
chat_model = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    stop=[]
)

# ----- Webcam -----
class WebcamStream:
    """Thread-safe Webcam-Stream mit Hintergrund-Updates."""
    
    def __init__(self, camera_index: int = CAMERA_INDEX):
        self.stream = VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            self.stream = VideoCapture(camera_index)
        
        ok, frame = self.stream.read()
        if not ok:
            raise RuntimeError(f"Kamera nicht lesbar (Index {camera_index}).")
        
        self.frame = frame
        self.running = False
        self.lock = Lock()
        self.thread = None

    def start(self):
        """Startet Hintergrund-Thread für Frame-Updates."""
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        """Hintergrund-Thread: Liest Frames kontinuierlich."""
        fail_count = 0
        while self.running:
            ok, frame = self.stream.read()
            if not ok:
                fail_count += 1
                if fail_count > 30:
                    print("[Webcam] Zu viele Fehler – stoppt.")
                    self.running = False
                    break
                continue
            
            fail_count = 0
            with self.lock:
                self.frame = frame

    def read(self, encode: bool = False):
        """
        Liest aktuelles Frame.
        
        Args:
            encode: Wenn True, gibt Base64-JPEG zurück (für LLM Vision).
        """
        if not self.running:
            return None
        
        with self.lock:
            frame = self.frame.copy()
        
        if encode:
            _, buf = imencode(".jpeg", frame)
            return base64.b64encode(buf).decode("utf-8")
        return frame

    def stop(self):
        """Stoppt Stream und Thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        self.stream.release()

# ----- Assistant (Agent mit Tools + Vision) -----
class Assistant:
    """Voice-Assistant mit LLM-Agent, Tools, TTS und Chat-History."""

    # Rate limiting settings
    MAX_REQUESTS_PER_MINUTE = 10
    MAX_INPUT_LENGTH = 500

    def __init__(self, model):
        self.agent_executor = None
        self.agent_executor_vision = None  # Separate agent for vision queries
        self.model = model
        self._piper = {}  # Dict für mehrere Piper-Modelle (pro Sprache)
        self.history = ChatMessageHistory()
        self.request_times: deque = deque(maxlen=self.MAX_REQUESTS_PER_MINUTE)
        self.stop_tts = False  # Flag to interrupt TTS playback
        self.tts_thread = None  # Thread for TTS playback
        self._update_agent()  # Initial Agent erstellen

    def interrupt_tts(self):
        """Stoppt die aktuelle TTS-Wiedergabe."""
        if self.tts_thread and self.tts_thread.is_alive():
            print("[TTS] Unterbrochen durch Benutzer")
            self.stop_tts = True
            # Wait for thread to finish
            self.tts_thread.join(timeout=1.0)

    def set_language(self, lang_code: str):
        """
        Ändert die aktuelle Sprache (Whisper + Response + TTS).

        Args:
            lang_code: 'ru' oder 'de'
        """
        global CURRENT_LANGUAGE
        if lang_code not in LANGUAGE_CONFIG:
            print(f"[ERROR] Unbekannte Sprache: {lang_code}")
            return

        CURRENT_LANGUAGE = lang_code
        self._update_agent()  # Agent mit neuer Sprache neu erstellen
        print(f"[Language] Gewechselt zu: {LANGUAGE_CONFIG[lang_code]['display_name']}")

    def _update_agent(self):
        """Erstellt Agent mit aktueller Sprachkonfiguration neu."""
        self.agent_executor = self._create_agent(self.model, vision=False)
        self.agent_executor_vision = self._create_agent(self.model, vision=True)

    def _check_rate_limit(self) -> bool:
        """Check if request is within rate limit"""
        current_time = time.time()
        # Remove requests older than 60 seconds
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()

        # Check if we've exceeded the rate limit
        if len(self.request_times) >= self.MAX_REQUESTS_PER_MINUTE:
            return False

        # Add current request
        self.request_times.append(current_time)
        return True

    def _sanitize_input(self, user_input: str) -> Optional[str]:
        """
        Sanitize user input

        Args:
            user_input: Raw user input

        Returns:
            Sanitized input or None if invalid
        """
        # Strip whitespace
        sanitized = user_input.strip()

        # Check for empty input
        if not sanitized or len(sanitized) < 3:
            return None

        # Check length
        if len(sanitized) > self.MAX_INPUT_LENGTH:
            print(f"[WARNING] Input zu lang ({len(sanitized)} chars), gekürzt auf {self.MAX_INPUT_LENGTH}")
            sanitized = sanitized[:self.MAX_INPUT_LENGTH]

        # Remove any control characters except newlines and tabs
        sanitized = ''.join(
            char for char in sanitized
            if char.isprintable() or char in '\n\t'
        )

        return sanitized

    def _is_vision_query(self, prompt: str) -> bool:
        """
        Check if the user query is asking about what's visible in the camera.

        Args:
            prompt: User input text

        Returns:
            True if query is vision-related, False otherwise
        """
        prompt_lower = prompt.lower()

        # Vision keywords in different languages
        vision_keywords = [
            # English
            'what do you see', 'what can you see', 'describe what', 'what is visible',
            'what\'s in the image', 'what\'s on the image', 'describe the image',
            'what am i', 'who am i', 'what are you looking at', 'look at',

            # German
            'was siehst du', 'was kannst du sehen', 'beschreibe was', 'was ist sichtbar',
            'was ist auf dem bild', 'beschreibe das bild', 'was bin ich', 'wer bin ich',
            'schau dir', 'sieh dir',

            # Russian
            'что ты видишь', 'что видишь', 'что ты можешь видеть', 'опиши что',
            'что видно', 'что на изображении', 'опиши изображение', 'что я',
            'кто я', 'посмотри на', 'видно на картинке'
        ]

        return any(keyword in prompt_lower for keyword in vision_keywords)

    def answer(self, prompt: str, image_base64: str):
        """
        Verarbeitet User-Prompt und antwortet mit TTS.

        Args:
            prompt: User-Frage (von Whisper erkannt)
            image_base64: Base64-encodiertes Webcam-Bild
        """
        # Rate limiting check
        if not self._check_rate_limit():
            print("[WARNING] Rate limit erreicht, bitte warten...")
            self._tts("Bitte langsamer sprechen, zu viele Anfragen.")
            return

        # Sanitize input
        sanitized_prompt = self._sanitize_input(prompt)
        if not sanitized_prompt:
            return

        print(f"[User] {sanitized_prompt}")

        # Check if this is a vision query
        is_vision = self._is_vision_query(sanitized_prompt)
        if is_vision:
            print("[Vision] Query detected - using vision-enabled agent")

        try:
            # Use appropriate agent based on query type
            if is_vision:
                response = self.agent_executor_vision.invoke({
                    "input": sanitized_prompt,
                    "image_base64": image_base64,
                    "chat_history": self.history.messages,
                })
            else:
                response = self.agent_executor.invoke({
                    "input": sanitized_prompt,
                    "chat_history": self.history.messages,
                })

            answer = response.get("output", "").strip()
            if not answer:
                return

            print(f"[Assistant] {answer}")

            # Update Chat-History
            self.history.add_user_message(sanitized_prompt)
            self.history.add_ai_message(answer)
            
            # TTS-Ausgabe
            self._tts(answer)
            
        except Exception as e:
            print(f"[ERROR] Agent-Fehler: {e}")
            import traceback
            traceback.print_exc()

    def _tts(self, text: str):
        """Text-to-Speech mit Piper (GPU-beschleunigt) in separatem Thread."""
        if DISABLE_TTS or not text:
            return

        # Stop any currently running TTS
        if self.tts_thread and self.tts_thread.is_alive():
            self.stop_tts = True
            self.tts_thread.join(timeout=1.0)

        # Reset flag and start new TTS thread
        self.stop_tts = False
        self.tts_thread = Thread(target=self._tts_piper, args=(text,), daemon=True)
        self.tts_thread.start()

    def _tts_piper(self, text: str):
        """Piper TTS Synthese und Wiedergabe."""
        # Entferne Markdown-Formatierung für bessere TTS-Ausgabe
        text = clean_markdown_for_tts(text)

        lang_config = LANGUAGE_CONFIG[CURRENT_LANGUAGE]
        piper_model_path = lang_config["piper_model"]

        if not piper_model_path:
            print(f"[TTS] Kein Piper-Model für {CURRENT_LANGUAGE} konfiguriert.")
            return

        try:
            import piper
            
            # Model einmalig pro Sprache laden
            if CURRENT_LANGUAGE not in self._piper:
                print(f"[TTS] Lade Piper-Model ({CURRENT_LANGUAGE}): {piper_model_path}")
                self._piper[CURRENT_LANGUAGE] = piper.PiperVoice.load(
                    piper_model_path,
                    piper_model_path + ".json"
                )
            
            voice = self._piper[CURRENT_LANGUAGE]
            
            # Sample-Rate aus Config
            cfg = voice.config
            sample_rate = int(getattr(getattr(cfg, "audio", cfg), "sample_rate", 22050))
            
            # Audio generieren
            audio_bytes = b"".join(
                chunk.audio_int16_bytes 
                for chunk in voice.synthesize(text)
            )
            
            if not audio_bytes:
                print("[TTS] Kein Audio generiert.")
                return
            
            print(f"[TTS] Audio: {len(audio_bytes)} bytes, {sample_rate} Hz")

            # Wiedergabe via PyAudio in Chunks (ermöglicht Unterbrechung)
            p = PyAudio()
            stream = None
            try:
                # Audio-Ausgabegerät: Falls konfiguriert verwenden, sonst System-Standard
                stream_kwargs = {
                    "format": paInt16,
                    "channels": 1,
                    "rate": sample_rate,
                    "output": True
                }
                if AUDIO_OUTPUT_DEVICE is not None:
                    stream_kwargs["output_device_index"] = AUDIO_OUTPUT_DEVICE

                stream = p.open(**stream_kwargs)

                # Play audio in chunks to allow interruption
                chunk_size = 4096  # Bytes per chunk
                for i in range(0, len(audio_bytes), chunk_size):
                    if self.stop_tts:
                        print("[TTS] Wiedergabe gestoppt")
                        break
                    chunk = audio_bytes[i:i + chunk_size]
                    stream.write(chunk)

            finally:
                # Clean up resources
                if stream:
                    stream.stop_stream()
                    stream.close()
                p.terminate()

        except Exception as e:
            print(f"[TTS] Fehler: {e}")

    def _create_agent(self, model, vision: bool = False):
        """
        Erstellt Function-Calling-Agent mit Tools (optional mit Vision).

        Args:
            model: LLM model to use
            vision: If True, includes webcam image in prompt
        """
        today = datetime.now().strftime("%d.%m.%Y %H:%M")
        lang_config = LANGUAGE_CONFIG[CURRENT_LANGUAGE]
        response_lang = lang_config["response_lang"]

        if vision:
            system_message = f"""You are a helpful assistant. Answer all questions in {response_lang}.

**IMPORTANT CONTEXT:**
- TODAY'S DATE AND TIME: {today}
- Your training data is outdated (cutoff 2023). For current events, news, or time-sensitive information, you MUST use tools.
- When searching for news, ALWAYS include the current year (2025) or month in your search query.
- You have access to a webcam image. Describe what you see in the image based on the user's question.

Use the available tools when needed."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", [
                    {"type": "text", "text": "{input}"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_base64}"}},
                ]),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
        else:
            system_message = f"""You are a helpful assistant. Answer all questions in {response_lang}.

**IMPORTANT CONTEXT:**
- TODAY'S DATE AND TIME: {today}
- Your training data is outdated (cutoff 2023). For current events, news, or time-sensitive information, you MUST use tools.
- When searching for news, ALWAYS include the current year (2025) or month in your search query.

Use the available tools when needed."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

        agent = create_tool_calling_agent(llm=model, tools=DEFAULT_TOOLS, prompt=prompt)

        return AgentExecutor(
            agent=agent,
            tools=DEFAULT_TOOLS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
        )

# ----- Main Loop -----
def main():
    """Hauptschleife: Webcam + STT + Agent + TTS."""
    webcam_stream = WebcamStream().start()
    assistant = Assistant(chat_model)
    
    # Whisper STT Setup
    recognizer = Recognizer()

    # Mikrofon mit explizitem Device-Index (falls konfiguriert)
    if AUDIO_INPUT_DEVICE is not None:
        print(f"[STT] Verwende Mikrofon-Gerät: Index {AUDIO_INPUT_DEVICE}")
        microphone = Microphone(device_index=AUDIO_INPUT_DEVICE)
    else:
        print("[STT] Verwende Standard-Mikrofon")
        microphone = Microphone()

    with microphone as source:
        print("[STT] Kalibriere Mikrofon...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print(f"[STT] Energy Threshold nach Kalibrierung: {recognizer.energy_threshold}")

        # Setze einen höheren Mindest-Threshold, um Hintergrundgeräusche zu vermeiden
        if recognizer.energy_threshold < 300:
            print(f"[STT] WARNUNG: Energy Threshold sehr niedrig ({recognizer.energy_threshold})")
            print("[STT] Erhöhe auf Minimum 300 um Hintergrundgeräusche zu vermeiden")
            recognizer.energy_threshold = 300

        print(f"[STT] Bereit! Aktuelle Sprache: {LANGUAGE_CONFIG[CURRENT_LANGUAGE]['display_name']}")
        print(f"[STT] Energy Threshold: {recognizer.energy_threshold}")
    
    def audio_callback(recognizer, audio):
        """Background-Callback für Whisper STT."""
        try:
            lang_config = LANGUAGE_CONFIG[CURRENT_LANGUAGE]
            text = recognizer.recognize_whisper(
                audio,
                model=WHISPER_MODEL,
                language=lang_config["whisper_lang"]
            )
            assistant.answer(text, webcam_stream.read(encode=True))
        except UnknownValueError:
            pass  # Kein erkennbares Audio
        except Exception as e:
            print(f"[STT] Fehler: {e}")
    
    stop_listening = recognizer.listen_in_background(microphone, audio_callback)
    
    try:
        print("[Main] Hotkeys:")
        print("  [1] = Russian")
        print("  [2] = Deutsch")
        print("  [SPACE] = TTS stoppen")
        print("  [q/ESC] = Beenden")
        
        while True:
            frame = webcam_stream.read()
            if frame is None:
                print("[Main] Keine Frames – Abbruch.")
                break
            
            # Aktuelle Sprache oben links (groß)
            lang_display = LANGUAGE_CONFIG[CURRENT_LANGUAGE]['display_name']
            cv2.putText(frame, f"Language: {lang_display}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Hotkey-Info unten links (klein)
            hotkeys = [
                "[1] Russian",
                "[2] Deutsch",
                "[SPACE] Stop TTS",
                "[q] Exit"
            ]
            
            y_offset = frame.shape[0] - 105  # 105px vom unteren Rand (mehr Platz für 4 Zeilen)
            for i, line in enumerate(hotkeys):
                y_pos = y_offset + (i * 25)  # 25px Zeilenabstand
                cv2.putText(frame, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Alloy Voice Assistant", frame)
            
            key = cv2.waitKey(1)
            if key in [27, ord("q")]:  # ESC oder 'q'
                break
            elif key == ord("1"):  # Russisch
                assistant.set_language("ru")
            elif key == ord("2"):  # Deutsch
                assistant.set_language("de")
            elif key == ord(" "):  # Leertaste - TTS stoppen
                assistant.interrupt_tts()
                
    except KeyboardInterrupt:
        print("\n[Main] Beendet (Ctrl+C).")
    finally:
        stop_listening(wait_for_stop=False)
        webcam_stream.stop()
        cv2.destroyAllWindows()
        print("[Main] Cleanup abgeschlossen.")

if __name__ == "__main__":
    main()