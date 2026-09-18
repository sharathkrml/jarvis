from typing import Literal
from mlx_lm import load, generate
from pydantic import BaseModel

from config import env

MODEL_ID = env("LLM_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

_model, _tokenizer = None, None

def preload():
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = load(MODEL_ID)
    return _model, _tokenizer

def chat(messages: list[Message | dict], max_tokens=512):
    model, tokenizer = preload()
    prompt = tokenizer.apply_chat_template([m.model_dump() if isinstance(m, Message) else m for m in messages], tokenize=False, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

def main():
    print(chat([Message(role="system", content="Reply in one sentence."), Message(role="user", content="What can you do?")]))

if __name__ == "__main__":
    main()