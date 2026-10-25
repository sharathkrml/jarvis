.PHONY: help env install run devices stt llm tts config

FILE ?= test.wav
TEXT ?= Systems online. How can I help?

help:
	@printf "Available targets:\n"
	@printf "  make env              Create .env from .env.example if missing\n"
	@printf "  make install          Sync project dependencies with uv\n"
	@printf "  make run [DEVICE=1]   Start Jarvis, optionally selecting a microphone\n"
	@printf "  make devices         List audio devices\n"
	@printf "  make stt [FILE=...]  Transcribe a WAV file (default: test.wav)\n"
	@printf "  make llm             Run a one-shot LLM prompt\n"
	@printf "  make tts [TEXT=...]  Speak text with the configured TTS voice\n"
	@printf "  make config          Check environment configuration\n"

env:
	@test -f .env || cp .env.example .env

install:
	uv sync

run:
	uv run python main.py $(DEVICE)

devices:
	uv run python main.py --list

stt:
	uv run python stt.py "$(FILE)"

llm:
	uv run python llm.py

tts:
	uv run python tts.py "$(TEXT)"

config:
	uv run python config.py
