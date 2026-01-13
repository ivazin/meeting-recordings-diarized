# Meeting Diarization

A Python tool for transcribing and diarizing audio/video files, optimized for **Apple Silicon**.

It converts media files to text with speaker identification (diarization), handling format conversion automatically.

## Prerequisites

1.  **uv**: [Install uv](https://github.com/astral-sh/uv)
2.  **ffmpeg**: Install via Homebrew: `brew install ffmpeg`
3.  **Hugging Face Account**: You need an [access token](https://huggingface.co/settings/tokens).
    *   Accept terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).
    *   Accept terms for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).

## Setup

1.  Clone the repository.
2.  Create a `.env` file (or export variables) with your [Hugging Face token](https://huggingface.co/settings/tokens):
    ```bash
    HF_TOKEN=your_token_here
    ```
3.  Install dependencies:
    ```bash
    make install
    ```

## Configuration

You can configure the Whisper model used (e.g., to use unquantized models for higher precision) by setting the `WHISPER_MODEL` environment variable in your `.env` file or export it.

**Default (Fast, 4-bit quantized)**:
```bash
WHISPER_MODEL=mlx-community/whisper-large-v3-turbo
```

**Standard Large V3 (usually 4-bit)**:
```bash
WHISPER_MODEL=mlx-community/whisper-large-v3-mlx
```

**Unquantized Float16 (Best Quality, High RAM)**:
```bash
# Turbo variant (float16)
WHISPER_MODEL=mlx-community/whisper-large-v3-turbo-fp16

# Original V3 variant (float16)
WHISPER_MODEL=mlx-community/whisper-large-v3-fp16
```
## Usage

1.  Place your audio/video files (mp3, mkv, mov, wav, etc.) in the `input/` directory.
2.  Run the processing script:
    ```bash
    make run
    ```
3.  Find results in the `output/` directory, organized by filename.

## Output Structure

```
output/
  ├── my_video/
  │   ├── transcript.txt                  # Speaker-grouped text
  │   └── transcript_with_timestamps.txt  # Detailed timestamps
  └── interview/
      ├── transcript.txt
      └── transcript_with_timestamps.txt
```
