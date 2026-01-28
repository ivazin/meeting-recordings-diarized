.PHONY: run run-giga run-no-diarization install clean

INPUT_DIR ?= input

# Construct arguments
ARGS = --input_dir "$(INPUT_DIR)"
ifdef OUTPUT_DIR
    ARGS += --output_dir "$(OUTPUT_DIR)"
endif

run:
	uv run main.py $(ARGS)

run-giga:
	uv run main.py --model gigaam $(ARGS)

run-no-diarization:
	uv run main.py --no-diarization $(ARGS)

install:
	@echo "Installing dependencies with uv..."
	uv sync

clean:
	rm -rf output/
