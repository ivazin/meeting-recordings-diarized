.PHONY: run install clean

run:
	uv run main.py

install:
	@echo "Installing dependencies with uv..."
	uv sync

clean:
	rm -rf output/
