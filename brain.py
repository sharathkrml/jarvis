from mlx_lm import load, generate

def main():
    model, tokenizer = load("mlx-community/Qwen3-4B-Instruct-2507-4bit")
    print(generate(model, tokenizer, prompt="Reply in one sentence: what can you do?", max_tokens=120, verbose=False))

if __name__ == "__main__":
    main()