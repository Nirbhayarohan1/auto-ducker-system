import pyaudiowpatch as pyaudio
import numpy as np
import librosa
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import collections
import os
import logging
import threading

# ============================================================
#   SPOTIFY DEVELOPER CREDENTIALS — set these as env vars
#   export SPOTIFY_CLIENT_ID=...
#   export SPOTIFY_CLIENT_SECRET=...
# ============================================================
CLIENT_ID     = os.environ.get("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI  = "http://127.0.0.1:8888/callback"

if not CLIENT_ID or not CLIENT_SECRET:
    raise EnvironmentError(
        "Missing Spotify credentials. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET as environment variables."
    )

# --- Settings (tune these without touching logic) ---
SILENCE_DURATION   = 3.0
LOWERED_VOLUME     = 20
NORMAL_VOLUME      = 80
CHECK_INTERVAL     = 0.1
SPEECH_RATIO       = 2.5
BASELINE_SAMPLES   = 60
MIN_THRESHOLD      = 200

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def connect_spotify():
    """Connect to Spotify and verify at least one device is available."""
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope="user-modify-playback-state user-read-playback-state"
    ))

    devices = sp.devices().get("devices", [])
    if not devices:
        raise RuntimeError(
            "No active Spotify devices found. "
            "Open Spotify Web in your browser and start playing something first."
        )

    log.info("Available Spotify devices:")
    for d in devices:
        active_marker = " <-- active" if d["is_active"] else ""
        log.info(f"  [{d['type']}] {d['name']}{active_marker}")

    # Find the active device; fall back to the first available one
    active = next((d for d in devices if d["is_active"]), devices[0])
    log.info(f"Targeting device: {active['name']} (id={active['id']})")
    return sp, active["id"]


def set_volume(sp, device_id, volume):
    """Set Spotify volume on a specific device. Logs failures instead of swallowing them."""
    try:
        sp.volume(volume, device_id=device_id)
    except spotipy.exceptions.SpotifyException as e:
        log.warning(f"Spotify API error while setting volume: {e}")
    except Exception as e:
        log.error(f"Unexpected error setting volume: {e}")


def compute_vad_score(audio_data: np.ndarray, sr: int = 16000, n_fft: int = 512) -> float:
    """
    Compute VAD score using STFT-based spectral energy.
    
    Uses librosa.stft to compute frequency-domain energy for more accurate
    voice activity detection than simple RMS.
    
    Args:
        audio_data: int16 audio samples
        sr: sample rate in Hz (default 16000)
        n_fft: FFT size (default 512)
    
    Returns:
        Spectral energy score (higher = more likely speech)
    """
    try:
        if len(audio_data) == 0:
            return 0.0
        
        # Convert int16 to float32 normalized to [-1, 1]
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Compute STFT with parameters suitable for real-time VAD
        D = librosa.stft(audio_float, n_fft=n_fft, hop_length=n_fft//4, center=False)
        
        # Compute magnitude spectrogram
        magnitude = np.abs(D)
        
        # Sum energy across frequency bins for each frame, then average
        frame_energy = np.sum(magnitude, axis=0)
        spectral_energy = np.mean(frame_energy) if len(frame_energy) > 0 else 0.0
        
        return float(spectral_energy)
    except Exception as e:
        log.warning(f"VAD computation error: {e}")
        return 0.0


def read_mic_level(stream, chunk=1024) -> float:
    """Read mic VAD score using STFT-based spectral energy. Returns -1.0 on error."""
    try:
        data = stream.read(chunk, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        return compute_vad_score(audio_data)
    except Exception as e:
        log.warning(f"Mic read error: {e}")
        return -1.0


def mic_reader_thread(stream, result_holder, stop_event, chunk=1024):
    """Reads mic in a background thread to avoid blocking the main loop."""
    while not stop_event.is_set():
        level = read_mic_level(stream, chunk)
        result_holder["level"] = level
        time.sleep(0.05)


def run():
    log.info("Connecting to Spotify...")
    try:
        sp, device_id = connect_spotify()
    except RuntimeError as e:
        log.error(str(e))
        return
    log.info("Connected.\n")

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024
    )

    log.info("Calibrating ambient noise... (stay quiet for 2 seconds)")
    baseline_buffer = collections.deque(maxlen=BASELINE_SAMPLES)
    for _ in range(20):
        level = read_mic_level(stream)
        if level >= 0:
            baseline_buffer.append(level)
        time.sleep(0.1)

    # Start background mic reader so main loop never blocks on I/O
    stop_event   = threading.Event()
    result_holder = {"level": 0.0}
    reader = threading.Thread(
        target=mic_reader_thread,
        args=(stream, result_holder, stop_event),
        daemon=True
    )
    reader.start()

    log.info("Listening... (Ctrl+C to stop)")
    log.info(f"  Speech ratio  : {SPEECH_RATIO}x above ambient")
    log.info(f"  Duck volume   : {LOWERED_VOLUME}%")
    log.info(f"  Normal volume : {NORMAL_VOLUME}%")
    log.info(f"  Restore after : {SILENCE_DURATION}s of silence\n")

    volume_lowered   = False
    last_speech_time = time.time()

    try:
        while True:
            mic_level = result_holder["level"]

            # Ignore fault reads (-1) entirely
            if mic_level < 0:
                time.sleep(CHECK_INTERVAL)
                continue

            ambient   = np.mean(baseline_buffer) if baseline_buffer else MIN_THRESHOLD
            threshold = max(MIN_THRESHOLD, ambient * SPEECH_RATIO)

            currently_talking = mic_level > threshold

            if currently_talking:
                last_speech_time = time.time()
                if not volume_lowered:
                    log.info(
                        f"[TALKING] Ducking to {LOWERED_VOLUME}%  "
                        f"(mic={int(mic_level)} threshold={int(threshold)})"
                    )
                    set_volume(sp, device_id, LOWERED_VOLUME)
                    volume_lowered = True
            else:
                baseline_buffer.append(mic_level)
                silence_seconds = time.time() - last_speech_time
                if volume_lowered and silence_seconds >= SILENCE_DURATION:
                    log.info(f"[SILENT]  Restoring to {NORMAL_VOLUME}%")
                    set_volume(sp, device_id, NORMAL_VOLUME)
                    volume_lowered = False

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        log.info("Interrupted. Restoring Spotify volume...")

    except Exception as e:
        log.error(f"Unexpected crash: {e}. Attempting volume restore...")

    finally:
        # Always restore volume, even on crash
        set_volume(sp, device_id, NORMAL_VOLUME)
        stop_event.set()
        stream.stop_stream()
        stream.close()
        p.terminate()
        log.info("Done.")


if __name__ == "__main__":
    run()