"""
Offline Text-To-Speech (TTS) Engine and Speech Manager for ASL Fingerspelling Studio.
Provides asynchronous, non-blocking word pronunciation using native Windows SAPI,
pyttsx3, and safe multi-tier speech generation.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Offline Speech Synthesizer Backend.
    Uses native Windows SAPI COM (direct SpVoice/SpFileStream) and pyttsx3.
    Operates 100% offline with zero cloud dependency.
    """

    def __init__(self, voice_rate: int = 150, volume: int = 100) -> None:
        self.voice_rate = voice_rate
        self.volume = volume
        self._backend: str = "none"
        self._sapi_voice = None
        self._temp_dir = tempfile.gettempdir()
        self._init_backend()

    def _init_backend(self) -> None:
        """Initializes best available offline TTS backend."""
        # 1. Native Windows SAPI COM (Preferred on Windows: rock-solid, fast, offline)
        if os.name == "nt":
            try:
                import win32com.client
                import winsound
                # Test SAPI creation
                v = win32com.client.Dispatch("SAPI.SpVoice")
                # SAPI rate range is -10 to +10 (0 is normal)
                sapi_rate = int((self.voice_rate - 150) / 15)
                v.Rate = max(-10, min(10, sapi_rate))
                v.Volume = max(0, min(100, self.volume))
                self._sapi_voice = v
                self._backend = "sapi_native"
                logger.info("Initialized native Windows SAPI.SpVoice TTS engine.")
                return
            except Exception as e:
                logger.debug("Native SAPI initialization note: %s", e)

        # 2. pyttsx3 fallback
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate", self.voice_rate)
            eng.setProperty("volume", self.volume / 100.0)
            self._engine = eng
            self._backend = "pyttsx3"
            logger.info("Initialized pyttsx3 TTS engine.")
            return
        except Exception as e:
            logger.debug("pyttsx3 initialization note: %s", e)

        # 3. PowerShell System.Speech fallback
        if os.name == "nt":
            self._backend = "powershell_sapi"
            logger.info("Initialized PowerShell System.Speech fallback TTS engine.")
        else:
            self._backend = "mock"
            logger.info("Initialized Mock TTS engine.")

    @property
    def backend_name(self) -> str:
        return self._backend

    def synthesize_and_play(self, word: str) -> bool:
        """
        Synthesizes and plays pronunciation for the full word.
        Blocks only for the duration of the audio clip (executed within worker thread).
        """
        clean_word = "".join(c for c in word.strip() if c.isalnum() or c.isspace())
        if not clean_word:
            return False

        # Capitalize naturally for TTS pronunciation (e.g. 'KAJAL' -> 'Kajal')
        speech_text = clean_word.title()

        if self._backend == "sapi_native":
            return self._speak_sapi_native(speech_text)
        elif self._backend == "pyttsx3":
            return self._speak_pyttsx3(speech_text)
        elif self._backend == "powershell_sapi":
            return self._speak_powershell(speech_text)
        else:
            logger.debug("Mock TTS speaking: %s", speech_text)
            time.sleep(0.3)
            return True

    def _speak_sapi_native(self, text: str) -> bool:
        try:
            import win32com.client
            import winsound

            # Generate to local temp WAV file via SpFileStream to ensure clean sound device access
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            wav_path = os.path.join(self._temp_dir, f"asl_speech_{os.getpid()}.wav")

            stream.Open(wav_path, 3, False)  # 3 = SSFMCreateForWrite
            voice.AudioOutputStream = stream
            voice.Speak(text, 0)  # 0 = SVSFDefault (synchronous write to file)
            stream.Close()

            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                # Play WAV through Windows audio system (SND_SYNC waits for word to finish playing)
                winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
                return True
        except Exception as e:
            logger.debug("SAPI native audio error: %s, falling back to powershell", e)
            return self._speak_powershell(text)
        return False

    def _speak_pyttsx3(self, text: str) -> bool:
        try:
            self._engine.say(text)
            self._engine.runAndWait()
            return True
        except Exception as e:
            logger.debug("pyttsx3 audio error: %s, falling back to powershell", e)
            return self._speak_powershell(text)

    def _speak_powershell(self, text: str) -> bool:
        try:
            ps_cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=6,
            )
            return True
        except Exception as e:
            logger.debug("PowerShell TTS error: %s", e)
            return False


class SpeechManager:
    """
    Thread-safe, non-blocking asynchronous Speech Queue Manager.
    Ensures that webcam recognition loop is NEVER blocked by audio playback.
    Prevents overlapping speech and provides real-time playback status.
    """

    def __init__(self, enabled: bool = True, voice_rate: int = 150) -> None:
        self.enabled = enabled
        self.tts_engine = TTSEngine(voice_rate=voice_rate) if enabled else None
        self._speech_queue: queue.Queue[str] = queue.Queue()
        self._is_speaking: bool = False
        self._last_spoken_word: str = ""
        self._status_text: str = "IDLE"
        self._running: bool = True
        self._lock = threading.Lock()

        # Dedicated background worker thread
        self._worker_thread = threading.Thread(
            target=self._speech_worker_loop,
            daemon=True,
            name="SpeechManagerWorker",
        )
        self._worker_thread.start()

    def speak(self, word: str) -> None:
        """
        Enqueues a word to be pronounced.
        Non-blocking: returns immediately to keep webcam at 30+ FPS.
        """
        if not self.enabled or not word or not word.strip():
            return
        clean = word.strip().upper()
        self._speech_queue.put(clean)

    def repeat_last(self) -> None:
        """Re-speaks the last spoken word."""
        with self._lock:
            last = self._last_spoken_word
        if last:
            self.speak(last)

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._is_speaking

    @property
    def last_spoken_word(self) -> str:
        with self._lock:
            return self._last_spoken_word

    @property
    def status_text(self) -> str:
        with self._lock:
            return self._status_text

    def _speech_worker_loop(self) -> None:
        """Background worker: drains queue and pronounces words sequentially."""
        while self._running:
            try:
                word = self._speech_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with self._lock:
                self._is_speaking = True
                self._status_text = f"SPEAKING: {word}"

            try:
                if self.tts_engine:
                    self.tts_engine.synthesize_and_play(word)
                with self._lock:
                    self._last_spoken_word = word
            except Exception as e:
                logger.error("TTS playback error for '%s': %s", word, e)
                with self._lock:
                    self._status_text = "ERROR"
            finally:
                with self._lock:
                    self._is_speaking = False
                    self._status_text = "IDLE"
                self._speech_queue.task_done()

    def shutdown(self) -> None:
        """Stops worker thread cleanly."""
        self._running = False
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)

