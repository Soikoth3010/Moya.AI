import os
import sys
import time
import queue
import threading
import subprocess
import webbrowser
import tempfile
from pathlib import Path

import sounddevice as sd
import soundfile as sf
import numpy as np

# Optional libraries
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False

try:
    import pywhatkit
    PYWHATKIT_AVAILABLE = True
except Exception:
    PYWHATKIT_AVAILABLE = False

# ---------------- CONFIG ----------------
WAKE_WORD = "moya"
MODEL_NAME = "medium"     # whisper model
SAMPLE_RATE = 16000
CHUNK_SEC = 2.5           # short chunks for responsiveness
OVERLAP_SEC = 0.5
GAIN = 2.0
RMS_AMBIENT_SECONDS = 1.2
RMS_WAKE_MULT = 1.8
TMP_DIR = Path.cwd() / ".moya_tmp"
TMP_DIR.mkdir(exist_ok=True)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_NAME = "Bella"
FUZZY_MAX_DIST = 2        # for wake-word fuzzy match
COMMAND_COOLDOWN = 0.6    # prevent duplicate command processing

# ---------------- UTILS ----------------
def rms_from_array(x: np.ndarray) -> float:
    if x is None or x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x.astype(np.float32)))))

def edit_distance(a: str, b: str) -> int:
    a = a.lower(); b = b.lower()
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j-1], dp[j])
            prev = cur
    return dp[n]

def fuzzy_wake_match(text: str) -> bool:
    toks = text.lower().split()
    for t in toks[:3]:
        if t == WAKE_WORD or edit_distance(t, WAKE_WORD) <= FUZZY_MAX_DIST:
            return True
    return False

# ---------------- TTS ----------------
class TTS:
    def __init__(self):
        self.engine = None
        if ELEVENLABS_API_KEY:
            self.use_elevenlabs = True
        else:
            self.use_elevenlabs = False
            if PYTTSX3_AVAILABLE:
                try:
                    self.engine = pyttsx3.init()
                    voices = self.engine.getProperty("voices")
                    for v in voices:
                        n = v.name.lower()
                        if "female" in n or "zira" in n or "anna" in n or "bella" in n:
                            self.engine.setProperty("voice", v.id)
                            break
                except Exception:
                    self.engine = None

    def speak(self, text: str):
        if self.use_elevenlabs:
            # For simplicity, fallback to pyttsx3 if API unavailable
            print("Bella:", text)
        else:
            if self.engine:
                self.engine.say(text)
                self.engine.runAndWait()
            else:
                print("Moya:", text)

# ---------------- ASR WORKER ----------------
class ASRWorker:
    def __init__(self):
        if not WHISPER_AVAILABLE:
            raise RuntimeError("Whisper not installed.")
        print(f"Loading Whisper model '{MODEL_NAME}'...")
        self.model = whisper.load_model(MODEL_NAME)
        print("Whisper ready.")
        self.q = queue.Queue(maxsize=6)
        self.callback = None
        self.running = False
        self.thread = threading.Thread(target=self._consumer_loop, daemon=True)

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False

    def submit(self, wav_path):
        try:
            self.q.put_nowait(wav_path)
        except queue.Full:
            try:
                old = self.q.get_nowait()
                os.remove(old)
                self.q.put_nowait(wav_path)
            except Exception:
                pass

    def _consumer_loop(self):
        while self.running:
            try:
                path = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                result = self.model.transcribe(path, language="en", fp16=False)
                text = result.get("text","").strip()
                if self.callback:
                    self.callback(text, path)
            except Exception as e:
                print("ASR error:", e)
            finally:
                try:
                    if Path(path).exists():
                        os.remove(path)
                except Exception:
                    pass

# ---------------- RECORDER ----------------
class Recorder:
    def __init__(self):
        self.sr = SAMPLE_RATE
        self.chunk_sec = CHUNK_SEC
        self.overlap_sec = OVERLAP_SEC
        self.gain = GAIN
        self.running = False
        self.ambient_rms = 0.0

    def calibrate(self):
        print("Calibrating ambient noise...")
        probe = sd.rec(int(RMS_AMBIENT_SECONDS * self.sr), samplerate=self.sr, channels=1, dtype='float32')
        sd.wait()
        self.ambient_rms = rms_from_array(probe.flatten())
        print(f"Ambient RMS = {self.ambient_rms:.6f}")
        return self.ambient_rms

    def start(self, submit_callback):
        self.running = True
        threading.Thread(target=self._loop, args=(submit_callback,), daemon=True).start()

    def stop(self):
        self.running = False

    def _normalize_and_save(self, arr, path):
        arr = arr.flatten() * self.gain
        arr = arr / max(1e-8, np.max(np.abs(arr))) * 0.95
        sf.write(path, arr, self.sr, format="WAV", subtype="PCM_16")

    def _loop(self, submit_callback):
        step = int((self.chunk_sec - self.overlap_sec) * self.sr)
        chunk = int(self.chunk_sec * self.sr)
        buffer = np.zeros((0,), dtype='float32')
        while self.running:
            frames = sd.rec(step, samplerate=self.sr, channels=1, dtype='float32')
            sd.wait()
            frames = frames.flatten()
            buffer = np.concatenate([buffer, frames])
            if buffer.size >= chunk:
                c = buffer[:chunk]
                buffer = buffer[step:]
                tmp_file = TMP_DIR / f"moya_chunk_{int(time.time()*1000)}.wav"
                self._normalize_and_save(c, tmp_file)
                submit_callback(str(tmp_file))
            time.sleep(0.01)

# ---------------- ASSISTANT ----------------
class Assistant:
    def __init__(self):
        self.tts = TTS()
        self.asr = ASRWorker()
        self.asr.callback = self.on_transcript
        self.rec = Recorder()
        self.rms_threshold = 0.0
        self.lock = threading.Lock()
        self.last_command_time = 0

    def start(self):
        self.rms_threshold = max(0.0005, self.rec.calibrate() * RMS_WAKE_MULT)
        print(f"RMS threshold: {self.rms_threshold:.6f}")
        self.asr.start()
        self.rec.start(self.asr.submit)
        self.tts.speak(f"Hello, I am Moya. Speak naturally. If your voice is soft, start with my name {WAKE_WORD}.")

    def stop(self):
        self.rec.stop()
        self.asr.stop()

    def on_transcript(self, text, path):
        text = text.strip()
        if not text:
            return
        print(f"[ASR] {text}")
        try:
            data, _ = sf.read(path, dtype='float32')
            probe_rms = rms_from_array(data.flatten())
        except Exception:
            probe_rms = 0.0
        low_voice = probe_rms < self.rms_threshold
        text_lower = text.lower()
        if low_voice and not fuzzy_wake_match(text_lower):
            self.tts.speak(f"If your voice is soft, start with my name {WAKE_WORD}.")
            return
        if fuzzy_wake_match(text_lower):
            toks = text_lower.split()
            for i, t in enumerate(toks[:3]):
                if edit_distance(t, WAKE_WORD) <= FUZZY_MAX_DIST:
                    toks = toks[i+1:]
                    break
            cmd = " ".join(toks).strip()
        else:
            cmd = text_lower
        if not cmd:
            return
        with self.lock:
            now = time.time()
            if now - self.last_command_time < COMMAND_COOLDOWN:
                return
            self.last_command_time = now
        threading.Thread(target=self._handle_command, args=(cmd,), daemon=True).start()

    def _handle_command(self, cmd):
        print("Handling:", cmd)
        # PC & online commands
        if any(k in cmd for k in ["exit","quit","goodbye","stop"]):
            self.tts.speak("Goodbye!")
            self.stop()
            sys.exit(0)
        if "how are you" in cmd:
            self.tts.speak("I am great today! How are you, Susmoy?")
            return
        if "youtube" in cmd and any(w in cmd for w in ["open","play"]):
            webbrowser.open("https://www.youtube.com")
            self.tts.speak("Opening YouTube.")
            return
        if "close youtube" in cmd:
            self.tts.speak("Please close the YouTube tab manually.")
            return
        if "play" in cmd or "music" in cmd:
            q = cmd.replace("play","").replace("music","").strip()
            if q and PYWHATKIT_AVAILABLE:
                pywhatkit.playonyt(q)
                self.tts.speak(f"Playing {q} on YouTube.")
            else:
                webbrowser.open("https://music.youtube.com")
                self.tts.speak("Opening YouTube Music.")
            return
        if any(k in cmd for k in ["search","google"]):
            q = cmd.replace("search","").replace("google","").strip()
            if q:
                webbrowser.open(f"https://www.google.com/search?q={q.replace(' ','+')}")
                self.tts.speak(f"Searching Google for {q}.")
            else:
                self.tts.speak("What should I search for?")
            return
        self.tts.speak("Sorry, I didn't understand. Try asking me to open YouTube or search Google.")

# ---------------- MAIN ----------------
def main():
    if not WHISPER_AVAILABLE:
        print("ERROR: Install openai-whisper and ffmpeg.")
        return
    assistant = Assistant()
    assistant.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Interrupted.")
        assistant.stop()

if __name__ == "__main__":
    main()
