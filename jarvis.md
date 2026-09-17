# JARVIS.md — Local Voice Assistant on Mac M1 Pro 16GB (MLX + uv)

> M1 Pro 16GB, macOS 13+. MLX-only, no cloud. **uv only** — never `pip`, never system python 3.14 (breaks mlx wheels, use uv 3.11).
> Verdict: **MLX > Ollama** here — 15-30% faster, Python-native VAD→STT→LLM→TTS. Ollama MLX backend needs 32GB+.

## 0. Pipeline

```
Mic → VAD → STT (Whisper-mlx) → LLM (Qwen3-MLX + tools) → TTS (Kokoro-MLX) → Speaker
```

| Part | Model | Size |
|------|-------|------|
| LLM | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | ~2.5GB |
| LLM fallback | `mlx-community/Qwen3-1.7B-4bit` | ~1.1GB |
| STT | `mlx-community/whisper-small-mlx` | ~290MB |
| TTS | `mlx-community/Kokoro-82M-bf16` | ~300MB |
| VAD | Silero / webrtcvad | ~2MB |

Budget: ~5-7GB resident. macOS ~3.5GB + stack ~3.5GB = fits. Skip 7-8B for v1 (no headroom).
<!-- ponytail: dropped LFM2.5-1.2B, 9B/35B, KV-cache tuning — add when 4B measurably too slow -->

Clone first, build second: [joyecai/bandy](https://github.com/joyecai/bandy) (full-duplex template), [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) (STT+TTS lib).
<!-- ponytail: dropped helomi/rcli/m1k3/wyoming/peekaboo/localtalk table — re-add when bandy insufficient -->

## 1. Prereqs

```bash
sw_vers && system_profiler SPHardwareDataType | grep -E "Chip|Memory"
xcode-select --install
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
brew install ffmpeg portaudio  # portaudio builds pyaudio
# Mic: System Settings → Privacy → Microphone → allow Terminal
```

## 2. Env

```bash
cd ~/Documents/Github/jarvis
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install mlx-lm mlx-audio mlx-whisper sounddevice numpy scipy pyaudio "misaki[en]" webrtcvad-wheels
uv run python -c "import mlx.core as mx; print(mx.default_device())"  # expect Device(gpu, 0)
```
Rule: venv activated for everything below. New pkg = `uv pip install`. One-offs = `uvx --python 3.11 --from <pkg> <cli>`.

## 3. Smoke tests

```bash
source .venv/bin/activate
# LLM (~22-32 tok/s on 4B, ~35-45 on 1.7B — if laggy, stay on 1.7B)
mlx_lm.generate --model mlx-community/Qwen3-4B-Instruct-2507-4bit --prompt "Reply in one sentence: what can you do?" --max-tokens 120

# STT
ffmpeg -f avfoundation -i ":0" -t 5 -ar 16000 -ac 1 test.wav -y
mlx_whisper test.wav --model mlx-community/whisper-small-mlx --language en

# TTS
mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 --text "Hello, I am Jarvis." --voice af_heart --play
```

## 4. Loop — `jarvis.py`

VAD-gated record → STT → LLM → streaming TTS. Save as `jarvis.py`:

```python
"""JARVIS v1 — local voice loop on MLX (M1 Pro 16GB safe). Env: .venv (uv, py3.11)."""
import sounddevice as sd, numpy as np, queue, sys

# --- config ---
LLM_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"   # or Qwen3-1.7B-4bit for lower latency
STT_MODEL = "mlx-community/whisper-small-mlx"
TTS_MODEL = "mlx-community/Kokoro-82M-bf16"
SAMPLE_RATE = 16000
SILENCE_SECS = 0.8  # end-of-utterance

SYSTEM = ("You are Jarvis, a concise voice assistant running fully on-device. "
          "Reply in 1-2 short sentences, plain speech, no markdown, no lists. "
          "If a tool is needed, say what you're doing first.")

def record_utterance():
    """Block until speech → silence. Returns int16 mono audio."""
    import webrtcvad  # uv pip install webrtcvad-wheels
    vad = webrtcvad.Vad(2)
    q: queue.Queue = queue.Queue()
    buf, silent, started = [], 0.0, False
    frame_ms = 30
    frame_len = int(SAMPLE_RATE * frame_ms / 1000)
    def cb(indata, frames, time, status):
        q.put(indata.copy())
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="int16", blocksize=frame_len, callback=cb):
        while True:
            chunk = q.get().tobytes()
            speech = vad.is_speech(chunk, SAMPLE_RATE)
            if speech:
                started = True; silent = 0.0; buf.append(chunk)
            elif started:
                buf.append(chunk); silent += frame_ms / 1000
                if silent >= SILENCE_SECS:
                    break
    raw = b"".join(buf)
    return raw

def stt(raw: bytes) -> str:
    import tempfile, os
    import mlx_whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        import wave
        w = wave.open(f.name, "wb")
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
        w.writeframes(raw); w.close()
        out = mlx_whisper.transcribe(f.name, path_or_hf_repo=STT_MODEL, language="en")
        os.unlink(f.name)
        return out["text"].strip()

def llm_reply(history: list) -> str:
    from mlx_lm import load, generate
    model, tok = load(LLM_MODEL)
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}] + history,
        tokenize=False, add_generation_prompt=True)
    return generate(model, tok, prompt=prompt, max_tokens=150, verbose=False)

def speak(text: str):
    from mlx_audio.tts.utils import load_model
    if not hasattr(speak, "_m"):
        speak._m = load_model(TTS_MODEL)
    # sentence-stream: synthesize per sentence so first audio starts fast
    import re
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if not sent.strip():
            continue
        for r in speak._m.generate(text=sent, voice="af_heart", speed=1.0, lang_code="a"):
            sd.play(np.asarray(r.audio, dtype=np.float32), samplerate=24000)
            sd.wait()

def main():
    print("JARVIS listening — speak, pause, get reply. Ctrl+C to quit.")
    history = []
    while True:
        raw = record_utterance()
        heard = stt(raw)
        if not heard:
            continue
        print(f"\nYOU: {heard}")
        if heard.lower() in ("quit", "exit", "shutdown", "goodbye jarvis"):
            speak("Shutting down. Goodbye.")
            sys.exit(0)
        history.append({"role": "user", "content": heard})
        answer = llm_reply(history[-6:])  # sliding window: keep ctx small on 16GB
        print(f"JARVIS: {answer}")
        history.append({"role": "assistant", "content": answer})
        speak(answer)

if __name__ == "__main__":
    main()
```

```bash
uv run --project ~/Documents/Github/jarvis python jarvis.py
```
Knobs in order: `LLM_MODEL` → 1.7B if laggy; keep history ≤6 turns; `max_tokens` 100-150; TTS `-4bit` if memory yellow.

## 5. Tools

Qwen3 does native tool-calling. Minimal pattern:

```python
TOOLS = [
  {"type": "function", "function": {
    "name": "get_time", "description": "Current local time",
    "parameters": {"type": "object", "properties": {}}}},
  {"type": "function", "function": {
    "name": "run_shell", "description": "Run a safe read-only shell command",
    "parameters": {"type": "object",
      "properties": {"cmd": {"type": "string"}},
      "required": ["cmd"]}}},
]

def run_tool(name, args):
    if name == "get_time":
        from datetime import datetime
        return datetime.now().strftime("%H:%M")
    if name == "run_shell":
        import subprocess
        # ALLOWLIST — never raw shell from voice
        cmd = args.get("cmd", "")
        assert cmd.startswith(("ls ", "cat ", "date", "df -h", "uptime")), "blocked"
        return subprocess.check_output(cmd, shell=True, text=True)[:2000]
```

Pass `tools=TOOLS`, parse `<tool_call>`, execute, feed back as `role: tool`. Graduate to Qwen-Agent + MCP only when hand-rolled breaks.

## 6. Alternatives

- Demo today, no build: `speech-to-speech --local_mac_optimal_settings --llm_backend mlx-lm` or clone `bandy` (`python3 install.py`).
- Server mode: `mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8000`.
<!-- ponytail: dropped X/Reddit receipts §§9-10-12, LM Studio, Ollama hatch, GUI panels — re-add when debugging a specific model pick -->

*Machine: uv 0.6.0 / py 3.11 venv, ffmpeg 9.0.1 ✓, system py 3.14 — never use for mlx.*
