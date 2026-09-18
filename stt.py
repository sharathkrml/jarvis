import mlx.core as mx
import numpy as np
from mlx_audio.stt import load

from config import env


MODEL_ID = env("STT_MODEL", "mlx-community/parakeet-tdt-0.6b-v2")

_model = None


def _get():
    global _model
    if _model is None:
        _model = load(MODEL_ID)
    return _model


def preload():
    return _get()


def transcribe(audio="test.wav"):
    if isinstance(audio, np.ndarray):
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0
        audio = mx.array(audio)
    return _get().generate(audio).text


def main(path="test.wav"):
    print(transcribe(path))

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "test.wav")
