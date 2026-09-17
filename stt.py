import mlx_whisper
import numpy as np


def transcribe(audio="test.wav"):
    if isinstance(audio, np.ndarray) and audio.dtype != np.float32:
        audio = audio.astype(np.float32) / 32768.0
    return mlx_whisper.transcribe(audio, path_or_hf_repo="mlx-community/whisper-small-mlx", language="en")["text"]


def main(path="test.wav"):
    print(transcribe(path))

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "test.wav")