# JARVIS ✨

A voice assistant that lives on *your* Mac. No cloud. No API keys. No "your data is important to us" emails.
You talk, it listens, it yaps back. All local, all on-device. Built on MLX, runs on Apple Silicon.

```text
🎤 You ──▶ VAD ──▶ STT ─────────▶ LLM ──────────▶ TTS ─────────▶ 🔊 Speaker
          (hears   (parakeet      (Qwen3-4B        (Kokoro-82M
           you)     hears words)   thinks)          speaks)
```

---

## The vibe

1. **Listens** to you nonstop. VAD is the bouncer — it cuts the yapping when you go quiet for ~1 sec.
2. **Types out** what you said (Parakeet, on-device).
3. **Thinks** — a local Qwen3 model claps back in 1–2 sentences. Short and sweet, no essays.
4. **Speaks** it out loud (Kokoro TTS).
5. Rinse and repeat until you say `quit` / `exit` / `shutdown` / `goodbye jarvis` or hit `Ctrl+C`.

Looks like this:

```text
──────────────────── JARVIS ────────────────────
● Ready (16.2s, Ctrl+C to quit)
🎙 Listening…
╭──── You ────╮
│ abilities.  │
╰─────────────╯
╭──────────────── Jarvis (1.4s) ─────────────────╮
│ I can assist with tasks and answer questions…  │
╰────────────────────────────────────────────────╯
🎙 Listening…
```

---

## What you need

| Thing | The tea |
|-------|---------|
| Laptop | Any Apple Silicon Mac (M1+). Vibes tuned for M1 Pro 16GB. |
| macOS | 13 or newer. |
| Python | **3.11**. Do NOT use 3.14 — it beefs with `mlx` and breaks the whole thing. |
| Tool | [`uv`](https://astral.sh/uv/install.sh). Use `uv sync` / `uv run`, never plain old `pip`. |
| System stuff | `ffmpeg` + `portaudio` from Homebrew (portaudio handles mic/speaker). |
| Permission | Let your Terminal use the mic: System Settings → Privacy & Security → Microphone. |
| Storage | ~4GB of model downloads. One time only, pinky promise. |
| RAM | ~5–7GB for the models + macOS uses ~3.5GB. 16GB is comfy. |

Versions proven to work in this venv:

`mlx` 0.32.2 · `mlx-lm` 0.31.3 · `mlx-audio` 0.5.4 · `mlx-whisper` 0.4.3 · `misaki` 0.9.4 ·
`sounddevice` 0.5.6 · `webrtcvad-wheels` 2.0.14 · `rich` 15.0.0 · `pydantic` 2.13.5 ·
`numpy`/`scipy` 2.4.6/1.17.1 · `torch` 2.14.0

---

## Setup (like 5 mins)

```bash
# 1. Check your machine + build tools
sw_vers && system_profiler SPHardwareDataType | grep -E "Chip|Memory"
xcode-select --install

# 2. Get uv and the audio libs
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
brew install ffmpeg portaudio

# 3. Clone it, then install everything (Python 3.11, pls)
git clone https://github.com/sharathkrml/jarvis.git && cd jarvis
uv sync

# 4. Make sure GPU + mic actually show up
uv run python -c "import mlx.core as mx; print(mx.default_device())"   # should say Device(gpu, 0)
uv run python main.py --list
```

The model downloads happen automatically the first run and live in `~/.cache/huggingface`.

> If macOS never asks for mic access, go flip it on yourself: System Settings → Privacy & Security → Microphone, then restart your terminal.

---

## Running it

```bash
source .venv/bin/activate

python main.py          # your default mic
python main.py 1        # mic #1, if you have a million
python main.py --list   # show all your audio devices
```

How to not be confused:
- Talk, then **go quiet for ~1 sec**. That's how it knows you're done.
- Want out? Say **quit / exit / shutdown / goodbye jarvis**, or just `Ctrl+C`.
- See `…didn't catch that`? It heard noise but no words. Just repeat yourself king.

### Poke at each piece separately

```bash
python stt.py test.wav            # transcribe a wav file
python llm.py                     # ask the brain one thing
python tts.py "Systems online."   # make it say something
python config.py                  # check .env is loading
```

---

## Tweaking it

`config.py` reads your `.env` file (kept out of git) and drops the values into the environment.
Each module reads its model from there, with a sane default if you didn't set it.
So changing a model = editing `.env`. No code touched.

```dotenv
LLM_MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit   # or Qwen3-1.7B-4bit for speed
STT_MODEL=mlx-community/parakeet-tdt-0.6b-v2          # or whisper-small-mlx for tiny
TTS_MODEL=mlx-community/Kokoro-82M-bf16               # or Kokoro-82M-4bit if memory's tight
```

Other knobs, if you're feeling custom:

| Knob | Where | Default | What it does |
|------|-------|---------|--------------|
| `SILENCE_SECS` | `main.py` | `1.0` | How quiet before it decides you're done. Lower = snappier, might cut your pauses. |
| VAD level | `main.py` | `Vad(3)` | 0–3. Bump down to `2` in a quiet room. |
| `SYSTEM_PROMPT` | `main.py` | short, 1–2 sentences | Its whole personality. Keep "no markdown, no lists" or it'll read bullet points out loud. |
| `max_tokens` | `llm.py` | `512` | Max reply length. 100–150 = fast answers. |
| `voice` / `speed` / `lang_code` | `tts.py` | `af_heart` / `1.0` / `a` | Which voice, how fast, what language. |

> Laggy replies? Switch the LLM to 1.7B → drop `max_tokens` to 100–150 → try the 4-bit TTS. In that order.

---

## What's in the box

```text
jarvis/
├── main.py      # the voice loop + the pretty terminal UI
├── llm.py       # the brain (Qwen3 via mlx-lm)
├── stt.py       # the ears (Parakeet)
├── tts.py       # the mouth (Kokoro)
├── config.py    # loads .env, small helper
├── jarvis.md    # OG design notes + tool-calling ideas
├── README.md    # you're here
├── pyproject.toml    # the deps live here (commit it)
├── uv.lock           # exact locked versions (commit it)
├── .python-version   # pins Python 3.11 (commit it)
├── .env         # your model picks (git-ignored)
└── .venv/       # the venv, Python 3.11 (git-ignored)
```

Deps are declared in `pyproject.toml` and installed with `uv sync`. The other two
generated files should be committed too: `uv.lock` locks every version + hash so
you and CI get identical installs, and `.python-version` pins the interpreter so
nobody drifts off 3.11. Only libraries published to PyPI git-ignore `uv.lock`.

### The files in one breath

- **`main.py`** — loops forever. Feeds little 30ms chunks of audio to `webrtcvad`, waits for silence, then runs it through the whole chain. Also prints the nice panels.
- **`llm.py`** — loads Qwen3 once and keeps it. `chat(messages)` applies the chat template and returns its answer.
- **`stt.py`** — takes an audio array (or a file) and returns text. Handles the int16 → float32 conversion for you.
- **`tts.py`** — loads Kokoro, does a throwaway "hi" so the first real reply is instant, then plays back each chunk.
- **`config.py`** — reads `.env`, ignores comments and blanks, hands values to the rest of the app.

---

## The models

| Piece | Model | Size | Job |
|-------|-------|------|-----|
| LLM | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | ~2.5GB | The brains |
| LLM backup | `mlx-community/Qwen3-1.7B-4bit` | ~1.1GB | When 4B is too slow |
| STT | `mlx-community/parakeet-tdt-0.6b-v2` | ~1.2GB | Turns speech into text |
| STT backup | `mlx-community/whisper-small-mlx` | ~290MB | Smaller, faster |
| TTS | `mlx-community/Kokoro-82M-bf16` | ~300MB | The voice |
| VAD | `webrtcvad` | ~2MB | Detects when you stop talking |

---

## When stuff breaks

| It's broken | Fix |
|-------------|-----|
| Mic does nothing / instant `didn't catch that` | Turn on Terminal mic access, then `python main.py --list` and pass the right number. |
| `mlx` won't import | You're on Python 3.14 or used `pip`. Redo the venv with `uv venv --python 3.11`. |
| Replies are slow | Drop to `Qwen3-1.7B-4bit`, lower `max_tokens` to 100–150. |
| It interrupts you | Raise `SILENCE_SECS` to 1.2–1.5 or use `Vad(2)`. |
| PortAudio screams | `brew install portaudio`, then reinstall the deps. |
| Memory in shambles | Use the 1.7B LLM and/or `Kokoro-82M-4bit`. |
| Weird `torch.jit.script` warning | Upstream noise, already muted in `tts.py`. Ignore it. |
| `Creating new KokoroPipeline…` mid-convo | It's pre-warmed at startup; if it shows up you're on an old `tts.py`. |

---

## Later-ish (not built yet)

- [ ] Keep only the last ~6 turns of chat history so it doesn't hoard memory.
- [ ] Tool calling (let it check the time, run safe commands). Idea sketched in `jarvis.md` §5.
- [ ] Stream the TTS sentence-by-sentence so it talks before it's done thinking.
- [ ] Barge-in — let you interrupt it mid-sentence.
- [ ] Wake word, so it only wakes up when you say "Hey Jarvis".

---

*Built with uv + Python 3.11. Your system 3.14? Leave it alone, it hates mlx.*
