.PHONY: run run-giga run-no-diarization install clean

run:
	uv run main.py

run-giga:
	uv run main.py --model gigaam

run-no-diarization:
	uv run main.py --no-diarization

install:
	@echo "Installing dependencies with uv..."
	uv sync

clean:
	rm -rf output/
