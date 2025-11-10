from threading import Lock, Thread
import os
import base64
import cv2
import time
from collections import deque
from datetime import datetime
from typing import Optional
from openai import OpenAI
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent  # ← geändert
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
        self.model = model
        self._piper = {}  # Dict für mehrere Piper-Modelle (pro Sprache)
        self.history = ChatMessageHistory()
        self.request_times: deque = deque(maxlen=self.MAX_REQUESTS_PER_MINUTE)
        self._update_agent()  # Initial Agent erstellen

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
        self.agent_executor = self._create_agent(self.model)

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

        try:
            response = self.agent_executor.invoke({
                "input": sanitized_prompt,
                "image_base64": image_base64,
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
        """Text-to-Speech mit Piper (GPU-beschleunigt)."""
        if DISABLE_TTS or not text:
            return
        self._tts_piper(text)

    def _tts_piper(self, text: str):
        """Piper TTS Synthese und Wiedergabe."""
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
            
            # Wiedergabe via PyAudio
            p = PyAudio()
            stream = p.open(
                format=paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
                output_device_index=3  # TODO: aus .env laden
            )
            stream.write(audio_bytes)
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            print(f"[TTS] Fehler: {e}")

    def _create_agent(self, model):
        """Erstellt Function-Calling-Agent mit Tools + Vision."""
        today = datetime.now().strftime("%d.%m.%Y %H:%M")
        lang_config = LANGUAGE_CONFIG[CURRENT_LANGUAGE]
        response_lang = lang_config["response_lang"]
        
        system_message = f"""You are a helpful assistant. Answer all questions in {response_lang}.

**IMPORTANT CONTEXT:**
- TODAY'S DATE AND TIME: {today}
- Your training data is outdated (cutoff 2023). For current events, news, or time-sensitive information, you MUST use tools.
- When searching for news, ALWAYS include the current year (2025) or month in your search query.
- You have access to a webcam image in every message. Use it to provide context-aware answers when the user asks "What do you see?" or similar questions.

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
        
        agent = create_openai_functions_agent(llm=model, tools=DEFAULT_TOOLS, prompt=prompt)
        
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
    microphone = Microphone()
    
    with microphone as source:
        print("[STT] Kalibriere Mikrofon...")
        recognizer.adjust_for_ambient_noise(source)
        print(f"[STT] Bereit! Aktuelle Sprache: {LANGUAGE_CONFIG[CURRENT_LANGUAGE]['display_name']}")
    
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
                "[q] Exit"
            ]
            
            y_offset = frame.shape[0] - 80  # 80px vom unteren Rand
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
                
    except KeyboardInterrupt:
        print("\n[Main] Beendet (Ctrl+C).")
    finally:
        stop_listening(wait_for_stop=False)
        webcam_stream.stop()
        cv2.destroyAllWindows()
        print("[Main] Cleanup abgeschlossen.")

if __name__ == "__main__":
    main()