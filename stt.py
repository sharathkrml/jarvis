import numpy as np
import sounddevice as sd
from mlx_audio.tts.utils import load_model


def main(text="Systems online. How can I help?"):
    model = load_model("mlx-community/Kokoro-82M-bf16")  # or .../Kokoro-82M-4bit
    for chunk in model.generate(text=text,
                                voice="af_heart", speed=1.0, lang_code="a"):
        audio = np.asarray(chunk.audio, dtype=np.float32)
        sd.play(audio, samplerate=chunk.sample_rate)
        sd.wait()


if __name__ == "__main__":
    main("I want you to tell exactly what I say.")