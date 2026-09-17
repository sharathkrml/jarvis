# JARVIS.md — Local Voice Assistant on Mac M1 Pro 16GB (MLX + uv)

> Target machine: **MacBook Pro M1 Pro, 16GB unified memory, macOS 13+**
> Stack: **MLX-only** (STT + LLM + TTS all local). No cloud required.
> Python: **uv exclusively** — no `pip`, no `python -m venv`, no system python (3.14 breaks mlx wheels).
> Verdict from research: **MLX > Ollama** for this build — 15-30% faster decode, ~10% less RAM, Python-native so you can wire VAD → STT → LLM → TTS in one process. Ollama (`llama.cpp + Metal`) is fine for chat prototyping, but it can't do low-latency full-duplex voice as cleanly, and Ollama 0.19's MLX backend needs 32GB+ anyway.

## 0. What you're building

```
Mic ──> VAD (Silero) ──> STT (Whisper-mlx) ──> LLM (Qwen3-MLX + tools) ──> TTS (Kokoro-MLX) ──> Speaker
                              │                         │
                              │                         └──> tools: time, weather, files, shell, web, MCP
                              └──> wake-word (optional) + dashboard (optional)
```

End-to-end budget on M1 Pro 16GB: **~5-7GB resident**. That's why model choice matters:

| Part | Model | Size | Why |
|------|-------|------|-----|
| LLM (default) | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | ~2.5GB | best tool-calling at 4B, 32k ctx |
| LLM (low-latency fallback) | `mlx-community/Qwen3-1.7B-4bit` | ~1.1GB | what Bandy uses; interrupts faster |
| LLM (ultra-fast) | `mlx-community/LFM2.5-1.2B-Instruct-4bit` | ~0.8GB | sub-200ms, RCLI default class |
| STT | `mlx-community/whisper-small-mlx` | ~290MB | ~30x realtime on M1 |
| TTS | `mlx-community/Kokoro-82M-bf16` (or `-4bit`) | ~300MB / ~100MB | 54 voices, Apache-2.0 |
| VAD | Silero VAD | ~2MB | sentence boundary + barge-in |

Usable RAM math: macOS takes ~3.5GB → ~12.5GB left. LLM 2.5GB + STT 0.5GB + TTS 0.5GB + browser/IDE 4GB = ~7.5GB. Comfortable. A 7-8B Q4 (~5GB) also fits but leaves no headroom for vision or long context — skip it for v1.

### Reference projects (real builds, same hardware class)

Verified via agent-reach (Exa + GitHub + X timelines + Reddit threads — see §9–§10):

1. **joyecai/bandy** — full-duplex MLX voice assistant, Whisper + Qwen3 + Kokoro/Qwen3-TTS + web dashboard. `python3 install.py` one-shot installer. Best template to clone. https://github.com/joyecai/bandy
2. **stanislaw-glogowski/helomi-app** — local-first wake-word → VAD → STT → streamed LLM → TTS with Apple Voice Processing echo cancellation. `make init`. Best for full-duplex/interrupt handling. https://github.com/stanislaw-glogowski/helomi-app
3. **runanywhereai/rcli** — STT+LLM+TTS+VLM pipeline, 40 macOS voice actions + local RAG. Note: MetalRT engine needs M3+, M1/M2 falls back to llama.cpp. Good for tool/action ideas, not as M1 engine. https://github.com/runanywhereai/rcli
4. **Round-Tower/m1k3** — Swift + MLX-Swift, Qwen3-4B + Gemma-12B + MCP server on `127.0.0.1:4242`. Look here if you want Swift UI later.
5. **Blaizzy/mlx-audio** — the library you'll use for both STT and TTS. https://github.com/Blaizzy/mlx-audio
6. **rnorth/wyoming-mlx** — Whisper-streaming + Kokoro as Wyoming + OpenAI audio endpoints (STT `:10300`, TTS `:10200`). Pick this if you want Home Assistant integration instead of hand-rolled `jarvis.py`.

---

## 1. Prerequisites (15 min)

```bash
# 1. macOS + chip check
sw_vers && sysctl -n machdep.cpu.brand_string
system_profiler SPHardwareDataType | grep -E "Chip|Memory"

# 2. Xcode CLI tools (needed for audio helper builds)
xcode-select --install
# If brew ever complains about the Xcode license:
#   sudo xcodebuild -license accept

# 3. uv (the ONLY python manager in this guide — verified 0.6.0 on this machine)
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
# Managed pythons already available: 3.10 / 3.11 / 3.12 — we pin 3.11 for mlx wheels.
# System python3 here is 3.14: NEVER use it for mlx (wheels break). uv enforces 3.11 below.

# 4. System audio deps (ffmpeg already installed here: 9.0.1 + ffplay ✓)
brew install ffmpeg portaudio
# portaudio is required to build `pyaudio` (mic input). espeak-ng NOT needed for mlx-audio path.

# 5. Mic permission: System Settings → Privacy → Microphone → Terminal / VS Code → allow
```

## 2. Create the project env with uv (5 min)

```bash
mkdir -p ~/jarvis && cd ~/jarvis

# One venv for the whole project, pinned to uv's managed 3.11 (never system 3.14)
uv venv --python 3.11 .venv
source .venv/bin/activate

# All installs go through `uv pip` — never bare `pip`
uv pip install mlx-lm mlx-audio mlx-whisper
uv pip install sounddevice numpy scipy pyaudio
uv pip install misaki  # Kokoro text processing (English)
# uv pip install "misaki[zh]"  # add if you need Mandarin
# uv pip install "misaki[ja]"  # add if you need Japanese

uv run python -c "import mlx.core as mx; print(mx.default_device())"
# expect: Device(gpu, 0)
```

Rules for the rest of this guide:
- Every `python`, `mlx_lm.*`, `mlx_audio.*`, `mlx_whisper` command runs with `~/jarvis/.venv` activated (or prefixed `uv run --project ~/jarvis` — same env).
- Every new package = `uv pip install <pkg>` (activated) — never `pip install`, never `sudo`.
- One-off tools without touching the env: `uvx --python 3.11 --from <pkg> <cli>` (used once in §3).

If `uv pip install pyaudio` fails: `brew install portaudio` first (see §1), then retry. Do not fall back to system pip.

## 3. Smoke-test the LLM (10 min)

Start with the 1.7B (fast download, verifies MLX works), then graduate to 4B. Venv activated (`source ~/jarvis/.venv/bin/activate`):

```bash
cd ~/jarvis && source .venv/bin/activate

# -- fast check (~1.1GB download) --
mlx_lm.generate \
  --model mlx-community/Qwen3-1.7B-4bit \
  --prompt "You are Jarvis, a concise voice assistant. Reply in one sentence: what can you do?" \
  --max-tokens 120

# -- main brain (~2.5GB download) --
mlx_lm.generate \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --prompt "You are Jarvis. User says: 'open my notes and summarize today'. List the tool calls you'd make, then give a one-sentence voice reply." \
  --max-tokens 200
```

No-env alternative (zero setup check, same result — uvx fetches mlx-lm into its own cache):

```bash
uvx --python 3.11 --from mlx-lm mlx_lm.generate \
  --model mlx-community/Qwen3-1.7B-4bit \
  --prompt "Reply in five words: systems online." --max-tokens 20
```

Expected on M1 Pro: **1.7B ~35-45 tok/s, 4B ~22-32 tok/s**. If 4B feels laggy with browser + IDE open, stay on 1.7B for v1 — voice latency matters more than reasoning depth.

Optional OpenAI-compatible server (so any agent harness can call it):

```bash
cd ~/jarvis && source .venv/bin/activate
mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8000
# test in another terminal:
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"say hi in 5 words"}]}'
```

Ollama escape hatch (prototyping only, not voice — already installed here, unrelated to uv):
```bash
ollama run qwen3:4b
# or: ollama run llama3.2:3b
```

## 4. Smoke-test STT (10 min)

Venv activated:

```bash
cd ~/jarvis && source .venv/bin/activate

# record a 5s test clip (speak into mic)
ffmpeg -f avfoundation -i ":0" -t 5 -ar 16000 -ac 1 ~/jarvis/test.wav -y
# (if ":0" is wrong device: ffmpeg -f avfoundation -list_devices true -i "")

# transcribe with mlx-whisper
mlx_whisper ~/jarvis/test.wav \
  --model mlx-community/whisper-small-mlx \
  --language en

# --- or via mlx-audio API (same model family) ---
uv run --project ~/jarvis python - <<'EOF'
from mlx_audio.stt.generate import generate_
text = generate_(
  model="mlx-community/whisper-small-mlx",
  audio="~/jarvis/test.wav",
)
print("HEARD:", text)
EOF
```

Pass criteria: your sentence transcribed with <10% word error in a quiet room. If bad: check input device, move to `whisper-medium-mlx` (~800MB, only if RAM allows) or add noise suppression later.

## 5. Smoke-test TTS (10 min)

Venv activated:

```bash
cd ~/jarvis && source .venv/bin/activate

# CLI test — writes ./kokoro-output/*.wav and plays it
mlx_audio.tts.generate \
  --model mlx-community/Kokoro-82M-bf16 \
  --text "Hello, I am Jarvis. All inference is running locally on your Mac." \
  --voice af_heart \
  --output_path ./kokoro-output --join_audio --play

# low-RAM variant (if Activity Monitor pressure goes yellow):
# mlx_audio.tts.generate --model mlx-community/Kokoro-82M-4bit --text "Low memory mode." --voice af_heart --play
```

Python API (what your loop will use):

```python
from mlx_audio.tts.utils import load_model
model = load_model("mlx-community/Kokoro-82M-bf16")  # or .../Kokoro-82M-4bit
for chunk in model.generate(text="Systems online. How can I help?",
                            voice="af_heart", speed=1.0, lang_code="a"):
    play(chunk.audio)  # stream chunk-by-chunk, don't wait for full sentence
```

Voices to try: `af_heart` (default F), `am_adam` / `am_michael` (M). Language codes: `a`=US-EN, `b`=UK-EN, `z`=Mandarin (needs `misaki[zh]`).

## 6. Wire the loop — `jarvis.py` (30 min)

This is the minimal working assistant: **VAD-gated record → STT → LLM → streaming TTS**. Save as `~/jarvis/jarvis.py`:

```python
"""JARVIS v1 — local voice loop on MLX (M1 Pro 16GB safe). Env: ~/jarvis/.venv (uv, py3.11)."""
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

Install the one extra dep and run (uv only):

```bash
cd ~/jarvis && source .venv/bin/activate
uv pip install webrtcvad-wheels
uv run --project ~/jarvis python jarvis.py
# (activated venv also works: python jarvis.py — same interpreter)
```

Tuning knobs (in order of impact on 16GB):
1. `LLM_MODEL` → drop to `Qwen3-1.7B-4bit` if replies lag.
2. `history[-6:]` → keep at 4-6 turns; long history kills TTFT on M1.
3. `max_tokens=150` → cap at 100-150 for voice; long answers feel slow even at 30 tok/s.
4. `Kokoro-82M-4bit` instead of `-bf16` if Memory pressure goes yellow.
5. Close Chrome tabs / Xcode simulators while testing — unified memory is shared.

## 7. Give it hands — tools / agent (30 min)

Qwen3 has native tool-calling. Add this after v1 loop works. Minimal pattern (no framework needed):

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
        # ALLOWLIST for v1 — never give raw shell to voice input
        cmd = args.get("cmd", "")
        assert cmd.startswith(("ls ", "cat ", "date", "df -h", "uptime")), "blocked"
        return subprocess.check_output(cmd, shell=True, text=True)[:2000]
```

Wire: pass `tools=TOOLS` via `mlx_lm` chat template or `Qwen-Agent` (`uv pip install Qwen-Agent`), parse `<tool_call>` blocks, execute, feed result back as `role: tool`, then generate the spoken summary. When you outgrow hand-rolled tools, point Qwen-Agent at an MCP config (`mcp-server-time`, `mcp-server-fetch`) — same pattern Bandy uses for its cloud-agent fallback.

Ideas to copy from reference builds:
- **Bandy**: `bandy_config.yaml` model-switching + dashboard prompt editing — steal the config-file idea early. (Note: Bandy's `install.py` builds its own pip venv; leave it alone, don't force it onto uv.)
- **Helomi**: Apple Voice Processing (`avfaudio`) for barge-in (interrupt while speaking). `pyaudio` alone can't echo-cancel.
- **RCLI**: 40 macOS actions (open app, screenshot, RAG over `~/Documents`). Add one at a time behind allowlist.

## 8. Faster paths (if you don't want to build from zero)

| Want | Do |
|------|----|
| Working demo today | `git clone https://github.com/joyecai/bandy.git && cd bandy && python3 install.py` — pick Whisper-small + Qwen3-1.7B minimal set (its own installer/env, uv not involved) |
| Best interrupt handling to study | clone `helomi-app`, run `make init`, read its audio driver + VAD code |
| GUI model switching | LM Studio → download `mlx-community/Qwen3-4B-Instruct-2507-4bit` → built-in OpenAI server on `:1234` |
| Maintained voice glue (no hand-rolled loop) | HF `speech-to-speech`: `uv pip install "speech-to-speech[kokoro,whisper-mlx]"` then `speech-to-speech --local_mac_optimal_settings --llm_backend mlx-lm` |

## 9. Sources — X verified 2026-09-17 (using agent-reach, platform X via backend twitter-cli)

Auth: user-supplied `TWITTER_AUTH_TOKEN` + `TWITTER_CT0` in child-process env only (not stored in this file).
Results:

* `twitter search` — FAILED both before and after `pipx upgrade twitter-cli` (0.8.5): `Twitter API error 404`. Known unstable GraphQL endpoint per agent-reach retry chain. Fell back to stable commands.
* `twitter feed -n 5` — OK (`ok: true`), auth valid.
* `twitter user-posts @awnihannun -n 10` — OK. Key hits:
  - @awnihannun retweet @jmorgan (2026-08-16): "Ollama's latest inference on Apple Silicon is also built on MLX: https://ollama.com/blog/mlx. MLX is transformative."
  - @awnihannun retweet @redp314 (2026-08-10): "Got Meta's new Muse Glimmer 30B running on my MacBook (M3 Max, 96GB)... Fastest right now: Ollama's MLX engine (DFlash included) at ~29 tok/s. Tuned llama.cpp: ~21, Raw mlx-vlm: ~10" — confirms MLX > llama.cpp even inside Ollama.
  - @awnihannun (2026-09-03): "Qwen 27B dense at 105 tok/s output on an M5 Max" via Uzu speculative decoding — headroom signal, not M1-relevant but confirms direction.
  - @awnihannun retweets @zcbenz: MLX v0.32.1 / v0.32.2 "optimizations benefiting both decoding and prefilling" + mlx-lm as central model registry plan.
* `twitter user-posts @exolabs -n 10` — OK. Relevant: EXO Thunderbolt-5 RDMA Mac clusters (4x M5 Ultra → ~4.8TB/s aggregate), USB-C Mac↔DGX Spark kernel patch. Cluster-scale only — skip on 16GB, note for future.
* `twitter user-posts @redp314 -n 10` — OK. Qwen3.8-Flash-Next tuning on DGX Spark (57 tok/s, +32%), iPhone realtime transcription demo. Proves decode-tuning matters for agents, but hardware is datacenter-class.
* `twitter user-posts @blaizzy` — OK, empty (`data: []`, likely wrong handle — mlx-audio author not found by that handle).
* fetches via Jina/Exa to ground the X claims:
  - https://ollama.com/blog/mlx (preview): Ollama on Apple Silicon now built on MLX, NVFP4, shared-prefix cache. **Requires >32GB unified memory, Qwen3.5-35B-A3B only in preview** — i.e. not for your M1 Pro 16GB today. Reason this guide stays pure `mlx-lm`.
  - `rnorth/wyoming-mlx` — Apple-Silicon-native Whisper (WhisperLiveKit streaming) + Kokoro as Wyoming + OpenAI audio endpoints. Good alternative if you want Home Assistant integration: STT `:10300`, TTS `:10200`.
  - HF `speech-to-speech` pipeline (July 2026, powers 9k Reachy Minis): Silero VAD v5 → Parakeet TDT / Whisper-MLX → local LLM → Qwen3-TTS / Kokoro, `--local_mac_optimal_settings` + `--llm_backend mlx-lm` + `--model_name mlx-community/Qwen3-4B-Instruct-2507-bf16`. Use instead of hand-rolled loop if you want maintained glue.
  - `mlxserve.com/local-ai-assistant` (Loki) — wake-word + barge-in + Telegram + cron, all local. Steal UX ideas.

Takeaway for M1 Pro 16GB: X consensus (MLX lead + builders) is MLX-native, small quant models, decode speed = agent speed. Nothing on X contradicts the Qwen3-4B/1.7B + Whisper-small + Kokoro-82M pick in §0.

* **16GB warning**: never load two LLMs at once, never run Ollama + `mlx_lm.server` simultaneously (duplicates ~3GB caches), watch Activity Monitor → Memory Pressure. Sustained swap >2GB = drop a model size.
* Next verification step on your machine: paste output of `mlx_lm.generate` tok/s + `vm_stat` pressure and I'll tune the model pick.

## 10. Reddit field notes (using agent-reach, platform Reddit via backend OpenCLI — 2026-09-17)

Bridge status: `agent-reach doctor` reports OpenCLI bridge connected; `opencli reddit search/read` works. `rdt` CLI still not installed (not needed).

* `opencli reddit read p283d969` — FAILED: `EMPTY_RESULT / HTTP 404, returned no data`. ID not accessible (wrong/removed/private or needs full `r/sub/comments/` URL). Send the full URL if you meant a specific thread and I'll re-pull it.
* `opencli reddit search "local voice assistant Mac MLX Ollama"` → top hit:
  - **[r/Qwen_AI `1rk62qw`](https://www.reddit.com/r/Qwen_AI/comments/1rk62qw/)** by u/SnooWoofers7340 (score 24, 8 comments): "Real-time voice-to-voice with your LLM on a Mac Studio M1 Ultra 64GB, 100% local". Stack: `mlx_lm.server --model mlx-community/Qwen3.5-35B-A3B-4bit :8081` + Pipecat `:7860` (Silero VAD → MLX Whisper Large V3 Turbo Q4 → Qwen → Kokoro 82M) + Telegram/n8n 25-tool agent + Discord bot, exposed via Cloudflare Tunnel. Total footprint ~18.5GB. Comments confirm low-latency voice on PC is hard; Linux/CUDA untested by OP. Takeaway for you: same pipeline shape as §0, but their 35B brain needs 64GB — on 16GB swap 35B → Qwen3-4B/1.7B and Whisper-Large → Whisper-small, keep Pipecat + Silero + Kokoro as-is.
* `opencli reddit search "MLX Ollama Apple Silicon M1 16GB local LLM"` → top hit:
  - **[r/hermesagent `1uc7rw5`](https://www.reddit.com/r/hermesagent/comments/1uc7rw5/)** "Mac + MLX Megathread" by u/Jonathan_Rivera (score 226, 42 comments, updated June 2026). Directly relevant table:
    - 16GB row: "Qwen3.5-9B Q4_K_M or MLX 4-bit | llama.cpp or Ollama | practical floor for Hermes. Preserve RAM for context — use quantized KV cache." + "For 16GB: start at 64K context, not 128K."
    - llama.cpp: fastest TTFT, KV-cache quant control critical for 16GB; MTP is a NET LOSS on Metal (Qwen3.5-9B 25.3 → 19.3 tok/s).
    - MLX-LM: 20-30% faster generation; Qwen3.6-35B-A3B 4-bit 61.2 tok/s vs 16.7 dense 27B (M1 Max 64GB); bugs: tool_calls parser #1293, MTP 1-2 token #1292.
    - Ollama: MLX backend since v0.19 (2x decode, 1.6x prefill on M5 Max); bugs: KV leak #16698 (24→75GB on M4 Max 64GB), inter-prompt delay #16170.
    - Rapid-MLX (new): "2-4x faster than Ollama, 0.08s cached TTFT, 17 tool parsers" — strongest Mac tool-calling backend per thread. Worth a trial (`rapid-mlx serve`, OpenAI-compatible) if `mlx_lm.server` tool-calling misbehaves on Qwen3.5/3.6.
  - Adjustment to this guide from that thread: 9B Q4 is viable on 16GB for **text-only** agents, but for **voice (STT+LLM+TTS resident)** stay with §0's 4B/1.7B pick. If you try 9B, use quantized KV (`--cache-type-k q8_0 --cache-type-v q4_0` on llama.cpp) and 64K ctx max.

## 12. Builders who shipped it — social references (2026-09-17)

People with a *working* local voice assistant on a Mac, not think-pieces. Use these as proof the stack works and as code to steal from. (X via `twitter user-posts`, Reddit via `opencli reddit search/read`, repos via Exa — all pulled today.)

### X — working demos on timelines

1. **@kwindla (Pipecat co-founder, verified) — 2026-09-12: "Peekaboo" experimental macOS voice assistant** ([tweet](https://x.com/kwindla/status/2098880411279802391), 48 likes / 46 bookmarks / video demo)
   - "Audio transcription and wake word monitoring run locally. LLM and screen vision use cloud models by default, but of course you can swap in a local model/endpoint."
   - Working voice queries on video: *"Peekaboo, tell me when the build finishes"* (event-driven!), *"was that invite in Slack, email, WhatsApp, or iMessage?"* (cross-app memory search), *"did we do evals on boule, pancake, flatbread, or the dgx spark?"* (experiment tracking).
   - Architecture follow-up ([tweet](https://x.com/kwindla/status/2098880412869357723)): WebView + Pipecat JS client, **native macOS audio transport with echo cancellation**, audio/vision/screen/history/ui/shell workers over a bus — "more or less how all software is going to look."
   - Code: **https://github.com/pipecat-ai/peekaboo** (from [tweet](https://x.com/kwindla/status/2098882055404990517)). Clone this before writing your own loop — it solves mic-echo + window management, the two things §6 hand-waves.
2. **@kwindla — 2026-09-11: Pipecat STT benchmark** ([tweet](https://x.com/kwindla/status/2098528469416337679), 84 likes / 71 bookmarks): open-source suite measuring **semantic WER + time-to-final-segment** across 1,000 fragments. Calibration targets for your STT: P50 TTFS <300ms, P95 close to it (Muse Voice Transcribe scored P50 392ms — best new-model release they've seen). Repo: `pipecat-ai/stt-benchmark`. Run your Whisper-small against it before blaming the LLM for bad answers.
3. **@kwindla — 2026-09-16: multi-model voice loops** ([tweet](https://x.com/kwindla/status/2100063872208310587)): fast voice front-end + slow backend agents (coding/vision) + structured-data side loops (`UIWorker` pattern). Validates §7's design: keep the 4B voice loop dumb-fast, dispatch heavy tools to a second model async.
4. **@awnihannun retweet [@redp314 — 2026-08-10](https://x.com/redp314/status/2086930849522409530)** (in §9): Muse Glimmer 30B on M3 Max — Ollama MLX engine ~29 tok/s vs llama.cpp ~21. Proof MLX wins hold for real agent-serving, not just benchmarks.

### Reddit — working builds with community proof

5. **[r/agenticAI `1w1zcur`](https://www.reddit.com/r/agenticAI/comments/1w1zcur/) (17↑, 20 comments): "collecting Jarvis approaches" survey + builder replies.** Most useful comment thread for this project:
   - u/L0cut15 ships **l0cut15/hermes-voice-assistant** — Jarvis wakeword build for cheap DIY hardware, Hermes-agent compatible.
   - OP's shortlist for agents that *actually do things* (Gmail/Linear test, per u/Deep_Ad1959's challenge): **RedPlanetHQ/core** (30+ MCP integrations, temporal knowledge graph), **alexberardi/jarvis** (self-hosted, 24-community-package "Pantry", per-person speaker recognition), **PartyArty/true-jarvis** (dual-brain: speech-to-speech face + frontier-LLM subagents in parallel).
6. **[r/buildinpublic `1sduprv`](https://www.reddit.com/r/buildinpublic/comments/1sduprv/): CODEC by u/SnooWoofers7340** (same builder as [r/Qwen_AI `1rk62qw`](https://www.reddit.com/r/Qwen_AI/comments/1rk62qw/) in §10) — a *year* of building, working today: "Hey CODEC" wake, screen-see-and-click via local vision model (UI-TARS pixel coords, no accessibility APIs), hold-right-CMD dictation (local SuperWhisper replacement), live voice calls from phone, 50+ local skills, 5 security layers. The end-state vision for this guide's §7.
7. **[r/Qwen_AI `1rk62qw`](https://www.reddit.com/r/Qwen_AI/comments/1rk62qw/)** — §10. The Pipecat+MLX recipe (Silero → Whisper-Large-Turbo-Q4 → Qwen → Kokoro) that CODEC later replaced with its own WebSocket pipeline — i.e. start on Pipecat, outgrow it only when you must.

### GitHub — closest code to clone (all working, all Mac)

| Repo | Why it matters for M1 Pro 16GB | Run |
|------|-------------------------------|-----|
| **kwindla/macos-local-voice-agents** | THE reference: Silero + smart-turn v2 + **MLX Whisper + Gemma3n 4B + Kokoro**, **<800ms voice-to-voice on M-series**, serverless WebRTC, `HF_HUB_OFFLINE=1` startup tip | `server/bot.py` + `npm run dev` |
| **bschink/jarvis** | Same name, same stack (whisper.cpp + Qwen3.5 9B + Kokoro-ONNX + Qwen3-TTS router), **uv-managed, 174 tests, launchd services, menu-bar health monitor** — copy its service layout | `uv sync --group dev` |
| **sidhellman/localtalk** | Offline Apple-Silicon voice (Whisper + Gemma3/MLX + Kokoro), **`uv tool install localtalk` / `uvx localtalk` one-liner** — fastest working demo to try before building | `localtalk --kokoro-voice af_nova` |
| **dflynnwit/voice-loop** | Minimal loop with **barge-in that works**: Silero + SmartTurn v3 + WebRTC AEC3 echo cancel + Moonshine STT + Gemma4 + Kokoro, `SOUL.md` persona + `MEMORY.md` | `uv sync && uv run voice_loop_mac.py` |
| **maadhav-codes/ollama-vox** | Menubar app (mlx-whisper + Ollama + Kokoro) with latency stats panel — steal the status-panel idea | `uv run ollama-vox --setup` |
| **saiprabhakar/Pipecat-NSK-workshop-demo** | Full-local Pipecat workshop bot (**uv sync**, Ollama `llama3.2:3b`, native interrupt pipeline) — best commented starter | `uv run server/simple_bot.py -t webrtc` |
| **Xaden DEV guide** (dev.to, Mar 2026) | Production-tested numbers on Apple Silicon: warm loop **1.5–2.5s** (Whisper 300ms + Ollama 8B ~1s + Kokoro 300ms); key lesson: **Kokoro CLI-per-utterance ~9s → 300ms via ONNX + persistent server** — never shell out per sentence | `voice-chat-fast.sh --vad` |

Net: start from `kwindla/macos-local-voice-agents` (Pipecat + MLX) or `localtalk` (one-liner), graduate toward CODEC/peekaboo architecture (workers + local audio transport) — exactly the path §§6–§8 describe.

## 11. Local machine state (2026-09-17, verified)

* `uv 0.6.0`, managed pythons 3.10/3.11/3.12 present; system `python3` = 3.14.7 (do not use).
* `ffmpeg 9.0.1` + `ffplay` installed. `portaudio` NOT installed (needed for `pyaudio`) — `brew install portaudio` before §2. If brew fails on Xcode license, run `sudo xcodebuild -license accept` first.
* Ollama: only `nomic-embed-text` local (+ 2 cloud aliases). Ollama server running.
* LM Studio `~/.lmstudio/models/` empty.
* HF cache wiped (23GB freed, §11 history) — §3–§5 downloads refetch on first run.
* `~/jarvis/` does not exist yet — §2 creates it.

---
*Rewritten 2026-09-17: pip/venv → uv throughout. Update when Ollama MLX-backend supports <32GB or when you move to M3+.*
