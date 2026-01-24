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

1.  Clone the repository:
    ```bash
    git clone git@github.com:ivazin/meeting-recordings-diarized.git
    cd meeting-recordings-diarized
    ```
2.  Create a `.env` file (or export variables) with your [Hugging Face token](https://huggingface.co/settings/tokens):
    ```bash
    HF_TOKEN=your_token_here
    ```
3.  Install dependencies:
    ```bash
    make install
    ```

## Configuration
 
You can configure the transcription model using the `--model` argument or environment variables.
 
### Models
 
1.  **Whisper (Default)**: Uses Apple's MLX-optimized Whisper.
    ```bash
    make run
    # OR
    uv run main.py --model whisper
    ```
 
2.  **GigaAM-v3**: Uses SaluteDevices' GigaAM-v3 model (optimized for Russian).
    ```bash
    make run-giga
    # OR
    uv run main.py --model gigaam
    ```
 
### Whisper Configuration
 
You can configure the specific Whisper model architecture (e.g., quantisation) by setting the `WHISPER_MODEL` environment variable in your `.env` file or export it.
 
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
    make run       # For Whisper
    make run-giga  # For GigaAM
    ```
3.  Find results in the `output/` directory, organized by filename.
 
## Output Structure
 
```
output/
  ├── my_video/                               # Directory named after input file
  │   ├── transcript_diarized.txt             # Text grouped by speaker blocks
  │   ├── transcript_with_timestamps.txt      # Line-by-line text with timestamps & speaker IDs
  │   ├── plain_transcript.txt                # Raw transcription text only
  │   └── plain_transcript_with_timestamps.txt # Raw text with timestamps (no speakers)
  └── interview/
      ├── ...
```
