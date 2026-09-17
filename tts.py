import numpy as np
import sounddevice as sd
from mlx_audio.tts.utils import load_model


MODEL_ID = "mlx-community/Kokoro-82M-bf16"  # or .../Kokoro-82M-4bit

_model = None

def preload():
    global _model
    if _model is None:
        _model = load_model(MODEL_ID)
    return _model


def speak(text, voice="af_heart", speed=1.0, lang_code="a"):
    model = preload()
    for chunk in model.generate(text=text, voice=voice, speed=speed, lang_code=lang_code):
        audio = np.asarray(chunk.audio, dtype=np.float32)
        sd.play(audio, samplerate=chunk.sample_rate)
        sd.wait()


def main(text="Systems online. How can I help?"):
    speak(text)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "Systems online. How can I help?")