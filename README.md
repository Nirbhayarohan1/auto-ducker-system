# Auto Ducker System

Automatically lowers your Spotify volume when you start talking, and restores it when you stop. No manual pausing, no missed words.

---

## How It Works

The script continuously listens to your microphone. It uses STFT-based Voice Activity Detection (VAD) — via `librosa.stft` — to measure spectral energy rather than simple RMS. When your voice crosses a dynamic threshold above the ambient noise baseline, it calls the Spotify API to duck the volume. Once silence persists for a configurable duration, volume is restored.

A background thread handles mic I/O so the main loop never blocks.

---

## Project Structure

```
auto-ducker-system/
├── auto1/
│   ├── auto3.py        # Main script — VAD engine + Spotify control
│   └── test_vad.py     # VAD test suite against labelled audio clips
└── config.ini          # All tunable parameters (no code edits needed)
```

---

## Requirements

- Python 3.8+
- Windows (uses `pyaudiowpatch` for WASAPI loopback)
- An active Spotify session on a reachable device
- Spotify Developer credentials (free)

### Dependencies

```bash
pip install pyaudiowpatch numpy librosa spotipy
```

---

## Spotify Setup

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create an app — set the redirect URI to `http://127.0.0.1:8888/callback`
3. Copy your **Client ID** and **Client Secret**
4. Export them as environment variables:

```bash
# Windows (cmd)
set SPOTIFY_CLIENT_ID=your_client_id
set SPOTIFY_CLIENT_SECRET=your_client_secret

# Windows (PowerShell)
$env:SPOTIFY_CLIENT_ID="your_client_id"
$env:SPOTIFY_CLIENT_SECRET="your_client_secret"
```

---

## Running

```bash
cd auto1
python auto3.py
```

On first run, a browser window will open for Spotify OAuth. After that, it caches the token.

**Stay quiet for the first ~2 seconds** — the script calibrates ambient noise during startup.

---

## Configuration (`config.ini`)

All parameters live here. No touching the source code required.

### `[Audio]`

| Key | Default | Description |
|-----|---------|-------------|
| `sample_rate` | `16000` | Mic sample rate in Hz |

### `[Ducking]`

| Key | Default | Description |
|-----|---------|-------------|
| `silence_timeout_sec` | `3` | Seconds of silence before volume restores |

> The script-level constants in `auto3.py` (`LOWERED_VOLUME`, `NORMAL_VOLUME`, `SPEECH_RATIO`, etc.) currently take precedence over `config.ini`. Edit them directly at the top of `auto3.py` if you need to change ducking volumes or sensitivity.

**Key constants in `auto3.py`:**

| Constant | Default | Description |
|----------|---------|-------------|
| `SILENCE_DURATION` | `3.0` | Seconds of silence before restoring volume |
| `LOWERED_VOLUME` | `20` | Spotify volume % when ducked |
| `NORMAL_VOLUME` | `80` | Spotify volume % when not talking |
| `SPEECH_RATIO` | `2.5` | Multiplier above ambient to classify as speech |
| `BASELINE_SAMPLES` | `60` | Rolling window size for ambient calibration |
| `MIN_THRESHOLD` | `200` | Minimum VAD score to ever trigger ducking |

---

## Testing VAD

The `test_vad.py` script measures detection accuracy against your own labelled audio clips.

**Set up test clips:**

```
auto1/
└── test_clips/
    ├── speech/
    │   └── sample1.wav
    └── silence/
        └── ambient1.wav
```

Or label by filename: `*_speech.wav`, `*_silence.wav`, `*_noise.wav`.

**Run:**

```bash
python test_vad.py
```

Outputs a pass/fail table and overall accuracy. If accuracy is below 80%, tweak the threshold in `test_vad.py` (`threshold = 40.0`) to match your mic and environment.

---

## Troubleshooting

**"No active Spotify devices found"**
Open Spotify Web Player in your browser and start playing something before running the script.

**Volume not ducking**
Your ambient noise may be high. Lower `SPEECH_RATIO` (e.g., `2.0`) or `MIN_THRESHOLD` in `auto3.py`.

**Volume ducking too aggressively (keyboard noise, etc.)**
Raise `SPEECH_RATIO` or `MIN_THRESHOLD`.

**OAuth browser window keeps appearing**
Delete the cached `.cache` file in the project directory and re-authenticate.

---

## License

MIT
