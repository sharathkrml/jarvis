from typing import Literal
from mlx_lm import load, generate
from pydantic import BaseModel

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

_model, _tokenizer = None, None

def _get():
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = load("mlx-community/Qwen3-4B-Instruct-2507-4bit")
    return _model, _tokenizer

def preload():
    return _get()

def chat(messages: list[Message | dict], max_tokens=512):
    model, tokenizer = preload()
    prompt = tokenizer.apply_chat_template([m.model_dump() if isinstance(m, Message) else m for m in messages], tokenize=False, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

def main():
    print(chat([Message(role="system", content="Reply in one sentence."), Message(role="user", content="What can you do?")]))

if __name__ == "__main__":
    main()