import sounddevice as sd, numpy as np, queue, sys, time
import webrtcvad  # uv pip install webrtcvad-wheels
from rich.console import Console
from rich.panel import Panel
from stt import transcribe, preload as preload_stt
from brain import chat, preload as preload_brain
from tts import speak, preload as preload_tts

console = Console()

SILENCE_SECS = 1.0  # seconds of silence to consider the end of an utterance
SYSTEM_PROMPT = ("You are Jarvis, a concise voice assistant running fully on-device. "
          "Reply in 1-2 short sentences, plain speech, no markdown, no lists. ")

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

def record_utterance(device=None):
    vad = webrtcvad.Vad(3)
    q = queue.Queue()
    max_silent = int(SILENCE_SECS * 16000 / 480)
    frames, silent = [], 0
    with sd.InputStream(device=device, samplerate=16000, channels=1, dtype='int16', blocksize=480, callback=lambda indata, frames_, time, status: q.put(indata.copy())):
        try:
            while True:
                frame = q.get()
                if vad.is_speech(frame.tobytes(), 16000):
                    frames.append(frame)
                    silent = 0
                elif frames:
                    frames.append(frame)
                    silent += 1
                    if silent >= max_silent:
                        audio = np.concatenate(frames, axis=0).flatten()
                        return audio
        except KeyboardInterrupt:
            console.print("\n[dim]Stopped.[/dim]")
            return None


def main(device=None):
    if device in ("--list", "-l"):
        print(sd.query_devices())
        return
    if isinstance(device, str):
        try:
            device = int(device)
        except ValueError:
            pass
    console.rule("[bold cyan]JARVIS[/bold cyan]", style="cyan")
    with console.status("Loading models...", spinner="dots"):
        t0 = time.monotonic()
        preload_stt()
        console.log("STT ready")
        preload_brain()
        console.log("Brain ready")
        preload_tts()
        console.log("TTS ready")
    console.print(f"[bold green]● Ready[/bold green] [dim]({(time.monotonic() - t0):.1f}s, Ctrl+C to quit)[/dim]")
    while True:
        console.print("[dim]🎙 Listening…[/dim]")
        audio = record_utterance(device)
        if audio is None:
            break
        with console.status("Transcribing…", spinner="dots"):
            text = transcribe(audio).strip()
        if not text:
            console.print("[dim]…didn't catch that[/dim]")
            continue
        console.print(Panel(text, title="You", border_style="blue", expand=False))
        if text.lower().strip(" .!") in ("quit", "exit", "shutdown", "goodbye jarvis"):
            bye = "Shutting down. Goodbye."
            console.print(Panel(bye, title="Jarvis", border_style="green", expand=False))
            speak(bye)
            break
        messages.append({"role": "user", "content": text})
        with console.status("Thinking…", spinner="dots"):
            t0 = time.monotonic()
            response = chat(messages)
        messages.append({"role": "assistant", "content": response})
        console.print(Panel(response, title=f"Jarvis [dim]({(time.monotonic() - t0):.1f}s)[/dim]", border_style="green", expand=False))
        speak(response)
  

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)