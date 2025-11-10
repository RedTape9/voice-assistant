"""Mikrofondiagnose für Alloy Voice Assistant."""

import speech_recognition as sr
import pyaudio

def list_microphones():
    """Zeigt alle verfügbaren Mikrofone."""
    print("=== Verfügbare Mikrofone ===")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"  [{index}] {name}")
    print()

def test_ambient_noise(device_index=None):
    """Testet Energy Threshold und Ambient Noise."""
    recognizer = sr.Recognizer()

    if device_index is not None:
        print(f"=== Energy Threshold Test (Gerät {device_index}) ===")
        microphone = sr.Microphone(device_index=device_index)
    else:
        print("=== Energy Threshold Test (Standard-Gerät) ===")
        microphone = sr.Microphone()

    print(f"Standard Energy Threshold: {recognizer.energy_threshold}")
    print(f"Standard Dynamic Energy Threshold: {recognizer.dynamic_energy_threshold}")
    print()

    print("Kalibriere Mikrofon (2 Sekunden Stille)...")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print(f"Nach Kalibrierung: {recognizer.energy_threshold}")

        # Warnung bei sehr niedrigen Werten
        if recognizer.energy_threshold < 300:
            print("⚠ WARNUNG: Sehr niedriger Threshold - könnte Hintergrundgeräusche aufnehmen!")
            print("  → Eventuell 'Stereo Mix' statt echtes Mikrofon aktiv")
        print()

        print("Höre auf Audio (5 Sekunden)...")
        print("Sprechen Sie JETZT: 'Test Test Eins Zwei Drei'")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

    print(f"Audio aufgenommen: {len(audio.get_raw_data())} bytes")
    print()

    print("Versuche Spracherkennung mit Whisper...")
    try:
        text = recognizer.recognize_whisper(audio, model="base", language="de")
        print(f"✓ Erkannt: {text}")
        return text
    except sr.UnknownValueError:
        print("✗ Keine Sprache erkannt")
        return None
    except Exception as e:
        print(f"✗ Fehler: {e}")
        return None

def test_specific_device(device_index):
    """Testet ein spezifisches Gerät interaktiv."""
    print(f"\n{'='*60}")
    print(f"TEST: Gerät [{device_index}]")
    print('='*60)

    mic_names = sr.Microphone.list_microphone_names()
    if device_index < len(mic_names):
        print(f"Name: {mic_names[device_index]}")
    print()

    result = test_ambient_noise(device_index=device_index)

    if result:
        print(f"\n✓ ERFOLG: Gerät {device_index} funktioniert!")
    else:
        print(f"\n✗ FEHLER: Gerät {device_index} erkennt keine Sprache")

    return result is not None

def test_audio_devices():
    """Zeigt PyAudio Geräte."""
    print("=== PyAudio Geräte ===")
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"[{i}] {info['name']}")
        print(f"    Input Channels: {info['maxInputChannels']}")
        print(f"    Output Channels: {info['maxOutputChannels']}")
        print(f"    Default Sample Rate: {info['defaultSampleRate']}")
        if info['maxInputChannels'] > 0:
            print(f"    ★ INPUT DEVICE")
        if info['maxOutputChannels'] > 0:
            print(f"    ★ OUTPUT DEVICE")
    p.terminate()
    print()

if __name__ == "__main__":
    import sys

    # Check if user specified a device index to test
    if len(sys.argv) > 1:
        try:
            device_index = int(sys.argv[1])
            test_specific_device(device_index)
        except ValueError:
            print("Fehler: Bitte geben Sie eine Zahl als Geräte-Index an")
            print("Verwendung: python diagnose_microphone.py [device_index]")
            sys.exit(1)
    else:
        # Show all devices and recommend which to test
        list_microphones()
        print("\n" + "="*60)
        print("EMPFEHLUNG: Testen Sie Ihr Jabra Headset:")
        print("  python diagnose_microphone.py 2")
        print("="*60)
        print("\nOder führen Sie einen vollständigen Test aus:")
        print()

        # Test default device
        test_ambient_noise()
