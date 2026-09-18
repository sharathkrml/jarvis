import contextlib
import io
import warnings

import numpy as np
import sounddevice as sd
from mlx_audio.tts.utils import load_model

from config import env

warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)


MODEL_ID = env("TTS_MODEL", "mlx-community/Kokoro-82M-bf16")  # or .../Kokoro-82M-4bit

_model = None

def preload():
    global _model
    if _model is None:
        _model = load_model(MODEL_ID)
        # Warm up: KokoroPipeline is created lazily on first generate()
        # (and prints noise). Force it now so first real speak is fast + quiet.
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in _model.generate(text="hi", voice="af_heart", speed=1.0, lang_code="a"):
                break
    return _model


def speak(text, voice="af_heart", speed=1.0, lang_code="a"):
    model = preload()
    with contextlib.redirect_stdout(io.StringIO()):
        chunks = model.generate(text=text, voice=voice, speed=speed, lang_code=lang_code)
        for chunk in chunks:
            audio = np.asarray(chunk.audio, dtype=np.float32)
            sd.play(audio, samplerate=chunk.sample_rate)
            sd.wait()


def main(text="Systems online. How can I help?"):
    speak(text)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "Systems online. How can I help?")