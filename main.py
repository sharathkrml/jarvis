import sounddevice as sd, numpy as np, queue, sys
import webrtcvad  # uv pip install webrtcvad-wheels
from stt import transcribe, preload as preload_stt
from brain import chat, preload as preload_brain
from tts import speak, preload as preload_tts

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
            print("Stopped.")


def main(device=None):
    if device in ("--list", "-l"):
        print(sd.query_devices())
        return
    if isinstance(device, str):
        try:
            device = int(device)
        except ValueError:
            pass
    print("Loading models...", flush=True)
    preload_stt()
    preload_brain()
    preload_tts()
    print("Ready. Listening...")
    while True:
        audio = record_utterance(device)
        text = transcribe(audio).strip()
        if text:
            messages.append({"role": "user", "content": text})
            print(f"You said: {text}")
            response = chat(messages)
            messages.append({"role": "assistant", "content": response})
            print(f"Jarvis: {response}")
            speak(response)
  

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)