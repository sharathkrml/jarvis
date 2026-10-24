import contextlib
import io
import re
import warnings

import numpy as np
import ptts
import sounddevice as sd
from mlx_audio.tts.utils import load_model

from config import env

warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)


MODEL_ID = env("TTS_MODEL", "gradium/phonon-31m-en")  # or .../Kokoro-82M-4bit
PHONON_MODEL = MODEL_ID.removeprefix("hf://").startswith("gradium/phonon-")
DEFAULT_VOICE = env("TTS_VOICE", "default" if PHONON_MODEL else "af_heart")
PHONON_VOICE_ID = re.compile(r"[A-Za-z0-9_-]{16}")

_model = None


def preload():
    global _model
    if _model is None:
        if not PHONON_MODEL and PHONON_VOICE_ID.fullmatch(DEFAULT_VOICE):
            raise ValueError(
                f"TTS_VOICE={DEFAULT_VOICE!r} looks like a Phonon catalog ID, "
                f"but TTS_MODEL={MODEL_ID!r} selects a non-Phonon model. "
                "Environment variables override .env; update the active "
                "TTS_MODEL or run `unset TTS_MODEL` to use the .env value."
            )

        if PHONON_MODEL:
            model_id = MODEL_ID if MODEL_ID.startswith("hf://") else f"hf://{MODEL_ID}"
            _model = ptts.TTS(model_id, lang="en", quant="q8")
            with _model.stream("hi", voice=DEFAULT_VOICE) as audio:
                next(iter(audio), None)
        else:
            _model = load_model(MODEL_ID)
            # Kokoro creates its pipeline lazily on first generate().
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in _model.generate(
                    text="hi", voice=DEFAULT_VOICE, speed=1.0, lang_code="a"
                ):
                    break
    return _model


def speak(text, voice=DEFAULT_VOICE, speed=1.0, lang_code="a"):
    if PHONON_MODEL and (speed != 1.0 or lang_code != "a"):
        raise ValueError("speed and lang_code are Kokoro-only options.")

    model = preload()
    if PHONON_MODEL:
        with model.stream(text, voice=voice) as audio:
            with sd.OutputStream(
                samplerate=model.sample_rate, channels=1, dtype="float32"
            ) as output:
                for chunk in audio:
                    output.write(np.asarray(chunk, dtype=np.float32).reshape(-1, 1))
        return

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