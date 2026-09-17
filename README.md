# JARVIS — Local Voice Assistant for Apple Silicon

> A fully **on-device** voice assistant for Mac (Apple Silicon). Speak → it transcribes → thinks → talks back.
> No cloud, no API keys, no data leaves your machine. Built on **MLX** for the M1 Pro 16GB.

> **Status:** working v1 voice loop (`main.py`). Tested on M1 Pro / macOS 13+ / Python 3.11 via `uv`.

---

## Table of Contents

- [What it does](#what-it-does)
- [How it works (pipeline)](#how-it-works-pipeline)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Module reference](#module-reference)
- [Models](#models)
- [Terminal UI](#terminal-ui)
- [Performance & memory](#performance--memory)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

---

## What it does

1. **Listens** to your microphone continuously (voice-activity detection trims silence).
2. **Transcribes** what you said (Whisper, on-device).
3. **Thinks** — a local LLM (Qwen3) replies in 1–2 short spoken sentences.
4. **Speaks** the reply out loud (Kokoro TTS, on-device).
5. Loops forever until you say `quit` / `exit` / `shutdown` / `goodbye jarvis`, or press `Ctrl+C`.

Example session:

```text
──────────────────── JARVIS ────────────────────
● Ready (18.4s, Ctrl+C to quit)
🎙 Listening…
╭──── You ────╮
│ abilities.  │
╰─────────────╯
╭───────────────── Jarvis (1.4s) ─────────────────╮
│ I can assist with tasks and answer questions…   │
╰─────────────────────────────────────────────────╯
🎙 Listening…
```

---

## How it works (pipeline)

```text
┌─────┐   ┌─────┐   ┌────────────────┐   ┌──────────────┐   ┌───────────────┐   ┌─────────┐
│ Mic │──▶│ VAD │──▶│ STT (Whisper)  │──▶│ LLM (Qwen3)  │──▶│ TTS (Kokoro)  │──▶│ Speaker │
│ 16k │   │webrt│   │ whisper-small  │   │ 4B-Instruct  │   │ 82M, af_heart │   │ 24kHz   │
│ mono│   │ cvad│   │ via mlx-whisper│   │ via mlx-lm   │   │ via mlx-audio │   │         │
└─────┘   └─────┘   └────────────────┘   └──────────────┘   └───────────────┘   └─────────┘
              │              │                  │                   │
        30ms frames,   int16→float32,      chat history,      sentence-level
        1.0s silence   English             system prompt       synthesis + playback
        end-of-speech                       1–2 sentences
```

**File → stage mapping:**

| Stage | File | Key function |
|-------|------|--------------|
| VAD + loop + UI | `main.py` | `record_utterance()`, `main()` |
| Speech-to-text | `stt.py` | `transcribe(audio)`, `preload()` |
| LLM reasoning | `brain.py` | `chat(messages)`, `preload()` |
| Text-to-speech | `tts.py` | `speak(text)`, `preload()` |

Each module lazily loads its model once (`preload()`), caches it in a global, and exposes a `main()` so it can be tested standalone (see [Usage](#usage)).

---

## Features

- 🎙️ **Push-to-nothing voice loop** — just speak, pause ~1s, get an answer. VAD auto-detects end of utterance.
- 🧠 **Local LLM brain** — Qwen3-4B-Instruct (4-bit) with a concise voice-assistant system prompt.
- 🗣️ **Natural TTS voice** — Kokoro `af_heart` voice; warmed up at startup so the first reply has no pipeline-build hitch.
- 🖥️ **Rich terminal UI** — header rule, loading spinners with per-model readiness, `You`/`Jarvis` panels, per-reply latency, dim listening hints.
- 🔇 **Quiet by design** — upstream library print-noise (`Creating new KokoroPipeline…`) and the `torch.jit.script` FutureWarning are suppressed/warmed up in `tts.py`, so they never interrupt the conversation view.
- 🧪 **Independently testable modules** — run `python stt.py`, `python brain.py`, or `python tts.py "text"` to smoke-test each stage.
- 🎛️ **Mic selection** — pass a device index or list devices with `--list`.

---

## Requirements

| Requirement | Details |
|-------------|---------|
| Hardware | Apple Silicon Mac (M1 or newer). Tuned for M1 Pro 16GB. |
| OS | macOS 13+ |
| Python | **3.11** (via `uv`). Do **not** use system Python 3.14 — it breaks `mlx` wheels. |
| Package manager | [`uv`](https://astral.sh/uv/install.sh) — **only** `uv pip`, never `pip`. |
| System libs | `ffmpeg`, `portaudio` (via Homebrew; portaudio builds pyaudio/sounddevice mic access). |
| Mic permission | System Settings → Privacy & Security → Microphone → allow your Terminal. |
| Disk | ~4GB for model weights (downloaded on first run from Hugging Face). |
| RAM (resident) | ~5–7GB total (models) + macOS ~3.5GB — fits 16GB; see [Performance](#performance--memory). |

Installed dependency versions (known-good):

| Package | Version |
|---------|---------|
| `mlx` / `mlx-metal` | 0.32.2 |
| `mlx-lm` | 0.31.3 |
| `mlx-whisper` | 0.4.3 |
| `mlx-audio` | 0.5.4 |
| `misaki[en]` | 0.9.4 (Kokoro phonemizer) |
| `sounddevice` | 0.5.6 |
| `webrtcvad-wheels` | 2.0.14 |
| `rich` | 15.0.0 |
| `torch` | 2.14.0 |
| `pydantic` | 2.13.5 |
| `numpy` / `scipy` | 2.4.6 / 1.17.1 |

---

## Installation

```bash
# 1. System sanity check
sw_vers && system_profiler SPHardwareDataType | grep -E "Chip|Memory"
xcode-select --install   # if not already installed

# 2. Install uv (if missing) + system audio libs
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
brew install ffmpeg portaudio

# 3. Clone & create the venv (Python 3.11, NOT system 3.14)
cd ~/Documents/Github/jarvis
uv venv --python 3.11 .venv && source .venv/bin/activate

# 4. Install Python deps (uv only)
uv pip install mlx-lm mlx-audio mlx-whisper sounddevice numpy scipy \
  pyaudio "misaki[en]" webrtcvad-wheels rich pydantic

# 5. Verify the GPU + mic
uv run python -c "import mlx.core as mx; print(mx.default_device())"  # expect Device(gpu, 0)
uv run python main.py --list   # list audio devices, note your mic index
```

> **Mic permission:** on first run macOS will prompt for microphone access. If it doesn't (or silently fails), go to
> System Settings → Privacy & Security → Microphone and enable your terminal app, then restart the terminal.

Model weights (~4GB) download automatically from Hugging Face on first run and are cached in `~/.cache/huggingface`.

---

## Usage

### Main voice loop

```bash
source .venv/bin/activate
uv run --project ~/Documents/Github/jarvis python main.py          # default mic
uv run --project ~/Documents/Github/jarvis python main.py 1        # mic device index 1
uv run --project ~/Documents/Github/jarvis python main.py --list   # list devices (-l also works)
```

- Speak, then **pause ~1 second** — silence ends the utterance.
- Say **quit / exit / shutdown / goodbye jarvis** (or `Ctrl+C`) to stop.
- If it prints `…didn't catch that`, it heard noise but no speech — just speak again.

### Test each stage standalone

```bash
# STT: transcribe a wav file (16kHz mono works best)
uv run python stt.py test.wav

# Brain: one-shot LLM reply
uv run python brain.py

# TTS: speak text out loud
uv run python tts.py "Systems online. How can I help?"
uv run python tts.py   # uses the default greeting
```

### Smoke-test models directly (no app code)

```bash
source .venv/bin/activate

# LLM speed check (~22–32 tok/s on 4B; if laggy, drop to 1.7B — see Configuration)
mlx_lm.generate --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --prompt "Reply in one sentence: what can you do?" --max-tokens 120

# STT check (record 5s, then transcribe)
ffmpeg -f avfoundation -i ":0" -t 5 -ar 16000 -ac 1 test.wav -y
mlx_whisper test.wav --model mlx-community/whisper-small-mlx --language en

# TTS check
mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 \
  --text "Hello, I am Jarvis." --voice af_heart --play
```

---

## Configuration

All the knobs that matter, and where they live:

### `main.py`

| Knob | Default | Effect |
|------|---------|--------|
| `SILENCE_SECS` | `1.0` | Silence duration that ends an utterance. Lower (0.7–0.8) = snappier but cuts off pauses; higher = waits longer. |
| `SYSTEM_PROMPT` | concise, 1–2 sentences, plain speech | Personality + reply shape. Keep "no markdown, no lists" so TTS doesn't read formatting aloud. |
| VAD aggressiveness | `webrtcvad.Vad(3)` (max) | `0–3`. `3` = strictest (fewer false triggers, may clip soft speech). Drop to `2` in quiet rooms. |
| Audio format | 16kHz, mono, int16, 480-sample blocks (30ms) | Must match webrtcvad's supported frame sizes (10/20/30ms at 16kHz). |
| Quit phrases | `quit, exit, shutdown, goodbye jarvis` | Case-insensitive, trailing `.`/`!` stripped. |

### `brain.py`

| Knob | Default | Effect |
|------|---------|--------|
| Model | `mlx-community/Qwen3-4B-Instruct-2507-4bit` (~2.5GB) | Fallback for speed: `mlx-community/Qwen3-1.7B-4bit` (~1.1GB). |
| `max_tokens` | `512` | Cap reply length. 100–150 is snappier for voice; 512 allows longer answers. |

### `stt.py`

| Knob | Default | Effect |
|------|---------|--------|
| Model | `mlx-community/whisper-small-mlx` (~290MB) | Good accuracy/speed balance on M1 Pro. |
| Language | `"en"` | Fixed English. |

### `tts.py`

| Knob | Default | Effect |
|------|---------|--------|
| Model | `mlx-community/Kokoro-82M-bf16` (~300MB) | `-4bit` variant exists if memory pressure is high. |
| Voice | `af_heart` | Kokoro voice preset. |
| Speed | `1.0` | `>1` faster, `<1` slower. |
| `lang_code` | `"a"` | Kokoro language bucket (American English). |

> Tuning order if responses feel slow: **LLM model → 1.7B** first, then lower `max_tokens` to 100–150, then consider the 4-bit TTS variant. (See `jarvis.md` §4 for the original tuning notes.)

---

## Project structure

```text
jarvis/
├── main.py          # Voice loop: VAD record → STT → LLM → TTS + Rich terminal UI
├── brain.py         # LLM: loads Qwen3 via mlx-lm, chat() with HF chat template
├── stt.py           # STT: int16→float32 normalize, whisper-small transcription
├── tts.py           # TTS: Kokoro voice, pipeline warm-up, noise-suppressed speak()
├── jarvis.md        # Original design notes (pipeline, smoke tests, tool-calling sketch)
├── README.md        # This file
├── .env             # Local env (git-ignored; VSCode terminal auto-loads it)
├── .vscode/
│   └── settings.json  # Points VSCode terminal at .env
└── .venv/           # uv virtualenv, Python 3.11 (git-ignored)
```

There is no `pyproject.toml` / `requirements.txt` — dependencies are installed with `uv pip install` (see [Installation](#installation)). `__pycache__/`, `*.wav`, and `kokoro-output/` are git-ignored.

---

## Module reference

### `main.py` — the loop

- `record_utterance(device=None)` — opens a `sounddevice.InputStream`, feeds 30ms frames to `webrtcvad`, buffers from first speech until `SILENCE_SECS` of non-speech, returns flattened int16 audio (or `None` on `Ctrl+C`).
- `main(device=None)` — `--list`/`-l` prints devices; otherwise preloads all three models under a spinner, then loops: listen → transcribe → panel → quit-check → `chat()` → panel → `speak()`. Conversation history (`messages`) is a module-level list starting with the system prompt and grows unbounded (see [Roadmap](#roadmap) for the sliding-window note).
- `__main__` passes `sys.argv[1]` (device index or `--list`) through.

### `brain.py` — the brain

- `Message` (pydantic) — validates `{role: system|user|assistant, content}`.
- `_get()` / `preload()` — loads Qwen3 once via `mlx_lm.load`, caches globally.
- `chat(messages, max_tokens=512)` — accepts `Message` or plain dicts, renders the HF chat template with `add_generation_prompt=True`, calls `mlx_lm.generate(verbose=False)`, returns the reply string.

### `stt.py` — the ears

- `preload()` — warms Whisper with 1s of silence.
- `transcribe(audio)` — accepts an int16/float32 numpy array (normalizes to float32 `/32768.0`) or a wav path; calls `mlx_whisper.transcribe(..., language="en")`, returns the text string.

### `tts.py` — the voice

- `preload()` — `load_model()` once, then a throwaway `generate("hi")` warm-up so the lazily-built `KokoroPipeline` (and its one-time `print`) happens at startup, not mid-conversation.
- `speak(text, voice="af_heart", speed=1.0, lang_code="a")` — wraps `model.generate()` in `redirect_stdout` (silences the pipeline print) and plays each chunk via `sounddevice` at the chunk's sample rate, blocking with `sd.wait()`.

---

## Models

| Part | Hugging Face ID | Size | Role |
|------|-----------------|------|------|
| LLM | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | ~2.5GB | Reasoning + replies |
| LLM fallback | `mlx-community/Qwen3-1.7B-4bit` | ~1.1GB | Lower latency option |
| STT | `mlx-community/whisper-small-mlx` | ~290MB | Speech-to-text (English) |
| TTS | `mlx-community/Kokoro-82M-bf16` | ~300MB | Speech synthesis (`af_heart`) |
| VAD | `webrtcvad` (code, no weights) | ~2MB | Speech/silence detection |

Total resident: ~5–7GB including runtimes → fits M1 Pro 16GB alongside macOS (~3.5GB). Skip 7–8B LLMs for v1 — no VRAM headroom.

---

## Terminal UI

Rendered with [`rich`](https://github.com/Textualize/rich) (already a dependency — no new packages):

- `console.rule("JARVIS")` header on startup.
- `console.status(...)` spinners for **Loading models…**, **Transcribing…**, **Thinking…**, with `console.log("STT/Brain/TTS ready")` progress during load and total load time on the `● Ready` line.
- Dim `🎙 Listening…` hint each turn; dim `…didn't catch that` on empty transcription.
- `Panel("…", title="You", blue)` for your words; `Panel("…", title="Jarvis (1.4s)", green)` for replies, including LLM latency.
- Graceful `Ctrl+C` (`Stopped.` + clean exit instead of a traceback).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Creating new KokoroPipeline for language: a` printed mid-chat | One-time lazy pipeline build inside `mlx-audio` (`kokoro.py:_get_pipeline`) on first `generate()` | Already handled: `tts.preload()` warms it at startup and `speak()` suppresses stdout. If you still see it, you're on an old `tts.py` — pull latest. |
| `FutureWarning: torch.jit.script is deprecated` | Upstream `mlx-audio`/torch noise (stderr) | Already filtered in `tts.py` via `warnings.filterwarnings`. Harmless if it reappears after a dep upgrade. |
| Mic hears nothing / immediate `…didn't catch that` | macOS mic permission, or wrong device | System Settings → Privacy → Microphone → allow Terminal; then `python main.py --list` and pass the right index. |
| `mlx` import fails / wheel errors | Wrong Python (system 3.14) or `pip` used | Recreate venv with `uv venv --python 3.11`, reinstall with `uv pip install`. Never use system python or plain `pip`. |
| Replies laggy (>3s) | 4B model on a loaded machine | Switch `brain.py` model to `Qwen3-1.7B-4bit`, cut `max_tokens` to 100–150. |
| Cuts you off mid-sentence | `SILENCE_SECS` too low / VAD too aggressive | Raise `SILENCE_SECS` to 1.2–1.5, or drop `Vad(3)` → `Vad(2)`. |
| `PortAudio` / device errors | Missing system lib | `brew install portaudio`, reinstall venv deps. |
| Memory pressure (yellow) | All three models + macOS on 16GB | Use 1.7B LLM and/or Kokoro `-4bit`; keep reply history short (see Roadmap). |

---

## Roadmap

Ideas parked for later (from `jarvis.md` + known gaps) — none implemented yet:

- [ ] **Sliding history window** — `main.py` history grows unbounded; cap to last ~6 turns to bound memory/latency on 16GB.
- [ ] **Tool calling** — Qwen3 supports native function calls; sketch in `jarvis.md` §5 (`get_time`, allowlisted `run_shell`) with `<tool_call>` parsing and `role: tool` feedback. Graduate to Qwen-Agent + MCP only when hand-rolled breaks.
- [ ] **Streaming TTS** — synthesize per sentence so first audio starts before the full reply is generated.
- [ ] **Full-duplex / interruption** — barge-in while Jarvis speaks (see [bandy](https://github.com/joyecai/bandy) as a template; `mlx-audio` for STT+TTS lib).
- [ ] **Server mode** — `mlx_lm.server --model … --port 8000` for multi-client use.
- [ ] `requirements.txt` / `pyproject.toml` — pin the known-good versions above for reproducible installs.
- [ ] Wake-word activation ("Hey Jarvis") instead of always-listening VAD.

---

*Machine notes: uv 0.6.0 / py 3.11 venv, ffmpeg 9.0.1 ✓, system py 3.14 — never use for mlx.*
