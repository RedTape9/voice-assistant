# 🎙️ Alloy Voice Assistant

**Multimodaler Voice-Assistant** mit lokaler Speech-to-Text (Whisper), Vision (Webcam), LLM-Agent (GPT-4o/Ollama) und Text-to-Speech (Piper). Unterstützt Russisch und Deutsch mit Hotkey-Umschaltung.

---

## 🎯 Features

- ✅ **Lokale Speech-to-Text** (OpenAI Whisper)
- ✅ **Lokale Text-to-Speech** (Piper, GPU-beschleunigt)
- ✅ **Vision** (Webcam-Integration, erkennt Objekte/Szenen)
- ✅ **LLM-Agent** mit Tools:
  - 🌐 Web-Suche (DuckDuckGo, kostenlos)
  - 🌤️ Wetter (wttr.in API, kostenlos)
  - 🕐 Aktuelle Uhrzeit/Datum
  - 🧮 Taschenrechner (sicheres `eval`)
- ✅ **Mehrsprachig** (Russisch/Deutsch, live umschaltbar)
- ✅ **Chat-History** (Kontext über mehrere Fragen)
- ✅ **Offline-fähig** (mit Ollama statt OpenAI)

---

## 📋 Voraussetzungen

### Hardware
- **Windows 10/11** (64-bit) oder **Linux**
- **Webcam** (USB oder integriert)
- **Mikrofon** (USB oder integriert)
- **GPU** (optional, empfohlen): NVIDIA mit CUDA 11.8+

### Software
- **Python 3.10 oder 3.11** (3.12+ nicht getestet)
- **Git** (für Klonen des Repos)
- **CUDA Toolkit 11.8+** (optional, für GPU-Beschleunigung)

---

## 🚀 Installation

### 1. Repository klonen

```bash
git clone https://github.com/yourusername/alloy-voice-assistant.git
cd alloy-voice-assistant
```

---

### 2. Python Virtual Environment erstellen

**Git Bash / Linux:**
```bash
python -m venv .venv
source .venv/Scripts/activate  # Git Bash (Windows)
source .venv/bin/activate      # Linux/macOS
```

**PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Prüfen:**
```bash
which python  # sollte .venv/Scripts/python zeigen
```

---

### 3. Dependencies installieren

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Falls `pyaudio` Fehler wirft (Windows):**
```bash
pip install pipwin
pipwin install pyaudio
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install portaudio19-dev python3-pyaudio ffmpeg
pip install -r requirements.txt
```

---

### 4. TTS-Modelle herunterladen

#### **Russisch (Irina, weiblich)**
```bash
mkdir -p models/piper/ru/ru_RU/irina/medium
cd models/piper/ru/ru_RU/irina/medium

curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json

cd ../../../../..
```

#### **Deutsch (Eva_K, weiblich)**
```bash
mkdir -p models/piper/de/de_DE/eva_k/x_low
cd models/piper/de/de_DE/eva_k/x_low

curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/eva_k/x_low/de_DE-eva_k-x_low.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/eva_k/x_low/de_DE-eva_k-x_low.onnx.json

cd ../../../../..
```

**Alternative deutsche Stimmen:**
- **Thorsten (männlich)**: `de/de_DE/thorsten/medium`
- **Karlsson (männlich, tief)**: `de/de_DE/karlsson/low`

Alle Stimmen: https://huggingface.co/rhasspy/piper-voices/tree/main

---

### 5. `.env` Datei erstellen

```bash
cp .env.example .env
nano .env  # oder Editor deiner Wahl (VS Code, Notepad++)
```

**Minimal-Konfiguration:**
```properties
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

DEFAULT_LANGUAGE=ru
PIPER_MODEL_RU=models/piper/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx
PIPER_MODEL_DE=models/piper/de/de_DE/eva_k/x_low/de_DE-eva_k-x_low.onnx
WHISPER_MODEL=base
CAMERA_INDEX=0
DISABLE_TTS=0
```

**Wichtig:** Ersetze `OPENAI_API_KEY` mit deinem echten Schlüssel!

---

### 6. API-Key besorgen

#### **Option A: OpenAI (kostenpflichtig, beste Qualität)**
1. Gehe zu https://platform.openai.com/api-keys
2. Erstelle neuen API-Key
3. Kopiere in `.env` → `OPENAI_API_KEY`

#### **Option B: Lokale Alternative (kostenlos, Offline)**
Nutze **Ollama** (lokal, keine API-Keys nötig):

```bash
# 1. Ollama installieren: https://ollama.com/download
# 2. Model herunterladen
ollama pull llama3.2-vision

# 3. Server starten
ollama serve  # läuft auf Port 11434
```

**`.env` anpassen:**
```properties
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.2-vision
OPENAI_API_KEY=dummy  # beliebiger Wert
```

---

## 🎮 Nutzung

### Starten

```bash
source .venv/Scripts/activate  # Git Bash
python assistant.py
```

**Erwartete Ausgabe:**
```
============================================================
🎙️  Alloy Voice Assistant
============================================================
✅ Webcam gestartet (Index: 0)
✅ LLM-Agent geladen (Model: gpt-4o)
🎤 Kalibriere Mikrofon...
✅ STT bereit (Whisper: base)
🌍 Aktuelle Sprache: Русский
============================================================

📋 Hotkeys:
  [1] = Русский (Russian)
  [2] = Deutsch (German)
  [q] / [ESC] = Beenden
```

---

### Hotkeys (während Laufzeit)

| Taste | Aktion |
|-------|--------|
| **`1`** | Wechsel zu Russisch |
| **`2`** | Wechsel zu Deutsch |
| **`q`** / **ESC** | Beenden |

---

### Beispiel-Prompts

#### **Russisch**
- *"Привет, как дела?"* → Grundlegende Konversation
- *"Какая погода в Москве?"* → Nutzt Wetter-Tool
- *"Найди новости про искусственный интеллект 2025"* → Nutzt Web-Suche
- *"Что ты видишь на камере?"* → Beschreibt Webcam-Bild
- *"Который час?"* → Nutzt Zeit-Tool
- *"Сколько будет 15 умножить на 7?"* → Nutzt Rechner-Tool

#### **Deutsch**
- *"Hallo, wie geht es dir?"* → Grundlegende Konversation
- *"Wie ist das Wetter in Berlin?"* → Nutzt Wetter-Tool
- *"Suche nach KI-News 2025"* → Nutzt Web-Suche
- *"Was siehst du vor der Kamera?"* → Beschreibt Webcam-Bild
- *"Wie spät ist es?"* → Nutzt Zeit-Tool
- *"Rechne 144 geteilt durch 12"* → Nutzt Rechner-Tool

---

## 🛠️ Konfiguration

### Audio-Device finden

```bash
python list_audio_devices.py
```

**Output-Beispiel:**
```
============================================================
Available Audio Output Devices:
============================================================

[0] Microsoft Sound Mapper - Output
    Host API: MME
    Sample Rate: 44100 Hz
    Channels: 2 ← DEFAULT

[3] Lautsprecher (Realtek High Definition Audio)
    Host API: MME
    Sample Rate: 48000 Hz
    Channels: 2
============================================================
```

**In `.env` setzen:**
```properties
AUDIO_OUTPUT_DEVICE=3
```

---

### Webcam-Index finden

```bash
python -c "import cv2; [print(f'Camera {i}: Available') for i in range(10) if cv2.VideoCapture(i).isOpened()]"
```

**Output-Beispiel:**
```
Camera 0: Available
Camera 1: Available
```

**In `.env` setzen:**
```properties
CAMERA_INDEX=1  # für externe USB-Kamera
```

---

### Neue Sprache hinzufügen

1. **Piper-Model herunterladen** (https://huggingface.co/rhasspy/piper-voices)
2. **In `assistant.py` → `LANGUAGE_CONFIG` hinzufügen:**

```python
"en": {
    "whisper_lang": "en",
    "response_lang": "English",
    "piper_model": "models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "display_name": "English"
}
```

3. **In `.env` definieren:**
```properties
PIPER_MODEL_EN=models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx
```

4. **Hotkey in `main()` hinzufügen:**
```python
elif key == ord("3"):  # Englisch
    assistant.set_language("en")
```

---

### Tools erweitern

**Beispiel: Wikipedia-Tool hinzufügen**

In `tools.py`:
```python
def wikipedia_tool(query: str) -> str:
    """Sucht in Wikipedia nach Informationen."""
    import wikipedia
    wikipedia.set_lang("de")  # oder "ru", "en"
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Wikipedia-Fehler: {e}"

DEFAULT_TOOLS.append(Tool(
    name="wikipedia",
    func=wikipedia_tool,
    description="Sucht in Wikipedia nach Fakten und Definitionen. Input: Suchbegriff."
))
```

---

## 🐛 Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'piper'`
**Lösung:**
```bash
pip install piper-tts~=1.3
```

---

### Problem: `pyaudio` Installation schlägt fehl
**Lösung (Windows):**
```bash
pip install pipwin
pipwin install pyaudio
```

**Lösung (Linux/Ubuntu):**
```bash
sudo apt install portaudio19-dev python3-pyaudio
pip install pyaudio
```

---

### Problem: Webcam wird nicht erkannt
**Lösung:**
```bash
# Verfügbare Kameras auflisten
python -c "import cv2; [print(f'Camera {i}: Available') for i in range(10) if cv2.VideoCapture(i).isOpened()]"

# Index in .env setzen
echo "CAMERA_INDEX=1" >> .env
```

---

### Problem: Kein Audio-Output (TTS stumm)
**Lösung:**
```bash
# Audio-Devices auflisten
python list_audio_devices.py

# Device in .env setzen (z.B. 3 für Lautsprecher)
echo "AUDIO_OUTPUT_DEVICE=3" >> .env
```

---

### Problem: `RuntimeWarning: Parameters {'stop'} should be specified`
**Status:** ✅ Bereits behoben in v1.0 (`stop=[]` in ChatOpenAI-Konstruktor)

---

### Problem: Agent nutzt veraltete Jahreszahl (2023)
**Status:** ✅ Bereits behoben – System-Prompt enthält: `TODAY'S DATE AND TIME: 19.10.2025`

---

### Problem: Vision funktioniert nicht
**Ursachen:**
1. Modell unterstützt keine Vision (z.B. `gpt-3.5-turbo`)
2. Webcam liefert keine Bilder

**Lösung:**
```bash
# 1. Prüfe Model in .env
OPENAI_MODEL=gpt-4o  # ✅ unterstützt Vision
# OPENAI_MODEL=gpt-3.5-turbo  # ❌ keine Vision

# 2. Teste Webcam
python -c "import cv2; cap=cv2.VideoCapture(0); print('OK' if cap.read()[0] else 'FEHLER')"
```

---

## 📦 Projekt-Struktur

```
alloy-voice-assistant/
├── assistant.py              # Hauptprogramm (Webcam + STT + Agent + TTS)
├── tools.py                  # Wiederverwendbare LangChain-Tools
├── list_audio_devices.py     # Helper: Audio-Devices auflisten
├── requirements.txt          # Python-Dependencies
├── .env.example              # Beispiel-Konfiguration
├── .env                      # Deine Konfiguration (GIT-IGNORED!)
├── .gitignore                # Git-Exclude-Rules
├── models/                   # TTS-Modelle (GIT-IGNORED!)
│   └── piper/
│       ├── ru/               # Russische Stimmen
│       └── de/               # Deutsche Stimmen
└── README.md                 # Diese Datei
```

---

## 🔒 Sicherheit

- ❌ **Committen Sie NIEMALS `.env`** (enthält API-Keys)
- ✅ `.env` ist bereits in `.gitignore`
- ✅ Nutzen Sie starke, einzigartige API-Keys
- ⚠️ `calculator`-Tool nutzt `eval()` → nur für vertrauenswürdige Inputs!
- 🔐 Empfehlung: Nutzen Sie separate API-Keys für Entwicklung/Production

---

## 📄 Lizenz

MIT License

```
MIT License

Copyright (c) 2025 [Dein Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Credits

- **OpenAI Whisper** (STT): https://github.com/openai/whisper
- **Piper TTS**: https://github.com/rhasspy/piper
- **LangChain**: https://python.langchain.com/
- **wttr.in** (Wetter-API): https://wttr.in/
- **DuckDuckGo Search**: https://pypi.org/project/ddgs/
- **OpenCV**: https://opencv.org/

---

## 🆘 Support

Bei Problemen:
1. ✅ Prüfe [Troubleshooting](#-troubleshooting)
2. 🐛 Öffne ein Issue auf GitHub
3. 📧 Kontakt: your.email@example.com

---

## 🚧 Roadmap

- [ ] **Englisch-Support** (in Arbeit)
- [ ] Wikipedia-Tool
- [ ] Kalender-Integration (Google Calendar)
- [ ] Smart-Home-Steuerung (Home Assistant API)
- [ ] Multi-User-Support (Stimmerkennung)
- [ ] Dockerisierung
- [ ] Web-UI (Flask/Gradio)
- [ ] Persistent Chat-History (SQLite)
- [ ] RAG-Integration (PDFs/Docs durchsuchen)

---

## 📊 Changelog

### v1.0.0 (2025-10-19)
- ✨ Initial Release
- ✅ Multimodal Agent (Text + Vision)
- ✅ Russisch/Deutsch-Support
- ✅ Lokale TTS (Piper)
- ✅ 4 Tools (Web, Wetter, Zeit, Rechner)

---

**Version:** 1.0.0 | **Letztes Update:** 19.10.2025