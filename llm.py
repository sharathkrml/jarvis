from typing import Literal
from mlx_lm import load, generate
from pydantic import BaseModel

from config import env
from tools import TOOLS, parse_calls, run as run_tool

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

def chat(messages: list[Message | dict], max_tokens=512, tools=None):
    model, tokenizer = preload()
    dicts = [m.model_dump() if isinstance(m, Message) else m for m in messages]
    prompt = tokenizer.apply_chat_template(
        dicts, tokenize=False, add_generation_prompt=True, tools=tools
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def chat_with_tools(messages: list[Message | dict], max_tokens=512, max_rounds=4, on_tool=None):
    """Answer, running any requested tools and feeding results back. Returns final text."""
    messages = list(messages)
    for _ in range(max_rounds):
        reply = chat(messages, max_tokens=max_tokens, tools=TOOLS)
        calls = parse_calls(reply)
        if not calls:
            return reply
        if "<tool_call>" in reply and "</tool_call>" not in reply:
            reply += "\n</tool_call>"  # ponytail: model drops close tag, template needs it
        messages.append({"role": "assistant", "content": reply})
        for name, args in calls:
            result = run_tool(name, args)
            if on_tool:
                on_tool(name, args, result)
            messages.append({"role": "tool", "content": result})
    return chat(messages, max_tokens=max_tokens)  # tool budget spent, just answer

def main():
    print(chat([Message(role="system", content="Reply in one sentence."), Message(role="user", content="What can you do?")]))

if __name__ == "__main__":
    main()