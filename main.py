import sounddevice as sd, numpy as np, queue, sys
import webrtcvad  # uv pip install webrtcvad-wheels
from stt import transcribe

SILENCE_SECS = 1.0  # seconds of silence to consider the end of an utterance
def record_utterance(device=None):
    vad = webrtcvad.Vad(3)
    q = queue.Queue()
    max_silent = int(SILENCE_SECS * 16000 / 480)
    frames, silent = [], 0
    with sd.InputStream(device=device, samplerate=16000, channels=1, dtype='int16', blocksize=480, callback=lambda indata, frames_, time, status: q.put(indata.copy())):
        print("Listening... speak, pause to transcribe. Ctrl+C to stop.")
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
                        frames, silent = [], 0
                        if len(audio) > 8000:
                            text = transcribe(audio).strip()
                            if text:
                                print(text)
                        else:
                            print("...")  # ponytail: 0.5s min, lower if short words get dropped
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
    record_utterance(device)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)