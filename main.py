import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import datetime

import mlx_whisper
from pyannote.audio import Pipeline
import torch
import warnings

# Filter specific warnings from dependencies that we can't control
warnings.filterwarnings("ignore", category=UserWarning, module="torch.load")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.load")
warnings.filterwarnings("ignore", message=".*weights_only=False.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
import subprocess


def load_env_file(env_path: Path):
    """Simple .env loader to avoid adding python-dotenv dependency."""
    if not env_path.exists():
        return
    
    logger.info(f"Loading environment variables from {env_path}")
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip("'").strip('"')

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS,mmm format."""
    td = datetime.timedelta(seconds=seconds)
    # Get total seconds for hours, minutes, seconds calculation
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

import time

def transcribe_audio(audio_path: str) -> Dict[str, Any]:
    """Transcribe audio using mlx-whisper."""
    logger.info("Starting transcription with mlx-whisper...")
    # MLX Whisper allows specifying a Hugging Face repo.
    # Default: "mlx-community/whisper-large-v3-turbo"
    # For float16 unquantized: "mlx-community/whisper-large-v3-turbo-fp16" or "mlx-community/whisper-large-v3-fp16"
    model_path = os.environ.get("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
    
    logger.info(f"Using Whisper model: {model_path}")
    
    start_time = time.time()
    result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model_path)
    end_time = time.time()
    
    duration = end_time - start_time
    logger.info(f"Transcription complete. Took {duration:.2f} seconds.")
    return result

def diarize_audio(audio_path: str, hf_token: str) -> Any:
    """Diarize audio using pyannote.audio."""
    logger.info("Starting diarization with pyannote.audio...")
    
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
    except Exception as e:
        logger.error(f"Failed to load pyannote pipeline: {e}")
        logger.error("Make sure you have accepted the user agreement on Hugging Face for pyannote/speaker-diarization-3.1")
        raise

    # Send pipeline to GPU (MPS) if available, though pyannote might default to CPU or CUDA
    # Pyannote support for MPS is experimental/limited, usually runs on CPU on Mac or needs specific setup.
    # We will try to use standard execution.
    if torch.backends.mps.is_available():
         pipeline.to(torch.device("mps"))
         logger.info("Using MPS for diarization.")
    else:
         logger.info("Using CPU for diarization.")

    # Run diarization
    diarization = pipeline(audio_path)
    logger.info("Diarization complete.")
    return diarization

def merge_results(transcription: Dict[str, Any], diarization: Any) -> List[Dict[str, Any]]:
    """Merge transcription segments with speaker diarization."""
    logger.info("Merging transcription and diarization results...")
    
    merged_segments = []
    
    # Iterate over whisper segments
    for seg in transcription['segments']:
        start = seg['start']
        end = seg['end']
        text = seg['text']
        
        # Find the speaker who speaks the most during this segment
        speakers_overlap = {}
        
        # diarization.itertracks(yield_label=True) yields (segment, track, label)
        # We need to query range.
        # Efficient way: loop through diarization segments intersecting with this text segment
        
        # Pyannote Annotation objects have a handy `crop` method but it might be strict.
        # We can just iterate linearly if not too huge, or use `crop`.
        
        # Let's use `crop` to get intersecting segments
        # crop returns a new Annotation containing only intersecting segments
        try:
            full_span = diarization.get_timeline().support() # total duration usually? No, this merges all.
            # actually we can just iterate.
            
            # Simple segment intersection
            segment_region = pyannote_segment(start, end)
            capturing_speakers = diarization.crop(segment_region)
            
            # Count duration for each speaker
            for turn, _, speaker in capturing_speakers.itertracks(yield_label=True):
                # Calculate overlap duration
                overlap_start = max(start, turn.start)
                overlap_end = min(end, turn.end)
                duration = max(0, overlap_end - overlap_start)
                
                if duration > 0:
                    speakers_overlap[speaker] = speakers_overlap.get(speaker, 0) + duration
            
        except Exception as e:
            # Fallback if I mess up pyannote API usage (it changes sometimes)
            # But crop is standard.
            # Need to import Segment from pyannote.core
            pass

        if speakers_overlap:
            # Pick speaker with max overlap
            best_speaker = max(speakers_overlap, key=speakers_overlap.get)
        else:
            best_speaker = "Unknown"
            
        merged_segments.append({
            "start": start,
            "end": end,
            "speaker": best_speaker,
            "text": text.strip()
        })
        
    return merged_segments

# Helper for pyannote segment, we need to import it or recreate it
from pyannote.core import Segment as pyannote_segment


def main():
    # 1. Configuration
    root_dir = Path(__file__).parent
    input_dir = root_dir / "input"
    output_base_dir = root_dir / "output"
    env_file = root_dir / ".env"
    
    # Load environment variables
    load_env_file(env_file)
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable not found. Please output it in .env or export it.")
        logger.error("You need a token with access to pyannote/speaker-diarization-3.1")
        return

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
        
    # Ensure output directory exists
    output_base_dir.mkdir(exist_ok=True)

    # List of extensions to process
    valid_extensions = {".mkv", ".mov", ".mp4", ".mp3", ".wav", ".m4a"}
    
    # Iterate over files in input directory
    for input_file in input_dir.iterdir():
        if input_file.is_dir():
            continue
            
        if input_file.suffix.lower() not in valid_extensions:
            # logger.info(f"Skipping unsupported file: {input_file.name}")
            continue
            
        # Check if this is a generated WAV file (heuristic: has same name as another file but .wav)
        # Or simpler: just process everything. But avoid re-processing the wav we just generated.
        # We will generate wavs in a temp location or side-by-side but we should pick *one* source per logical content.
        # For simplicity: process the file found. If we convert it to WAV, we use that wav for processing but results go to the original filename's folder.
        
        logger.info(f"Processing: {input_file.name}")
        
        # Create output directory for this specific file
        # Use name to create folder name (e.g. "my_video.mkv" from "my_video.mkv")
        file_output_dir = output_base_dir / input_file.name # input_file.stem for file name without extension
        file_output_dir.mkdir(exist_ok=True)

        # 2. Convert/Prepare Audio
        audio_file_to_process = input_file
        
        # Check if we need conversion (if not wav)
        if input_file.suffix.lower() != ".wav":
             try:
                 # Convert to WAV in the SAME input dir (as per previous logic) or temp?
                 # Let's keep it in input dir for now to avoid re-transcoding if run again? 
                 # Or better: put it in the output dir to keep input clean?
                 # User request: "batch process". 
                 # Let's convert to the file_output_dir to avoid cluttering input
                 
                 generated_wav = file_output_dir / f"{input_file.stem}.wav"
                 if not generated_wav.exists():
                     logger.info(f"Converting {input_file.name} to WAV in output dir...")
                     convert_to_wav_to_path(input_file, generated_wav)
                 else:
                     logger.info(f"Using existing WAV: {generated_wav}")
                 
                 audio_file_to_process = generated_wav
                 
             except Exception as e:
                 logger.error(f"Failed to convert {input_file.name}: {e}")
                 continue
        
        # 3. Transcription
        try:
            transcription_result = transcribe_audio(str(audio_file_to_process))
        except Exception as e:
            logger.exception(f"Transcription failed for {input_file.name}")
            continue

        # 4. Diarization
        try:
            diarization_result = diarize_audio(str(audio_file_to_process), hf_token)
        except Exception as e:
            logger.exception(f"Diarization failed for {input_file.name}")
            continue

        # 5. Merging
        final_segments = merge_results(transcription_result, diarization_result)
        
        # 6. Output
        output_txt = file_output_dir / "transcript.txt"
        output_ts = file_output_dir / "transcript_with_timestamps.txt"
        
        logger.info(f"Writing results to {output_txt} and {output_ts}")
        
        with open(output_txt, "w") as f_txt, open(output_ts, "w") as f_ts:
            current_speaker = None
            current_block = []
            
            for seg in final_segments:
                speaker = seg['speaker']
                text = seg['text']
                start_fmt = format_timestamp(seg['start'])
                end_fmt = format_timestamp(seg['end'])
                
                # Simple format: Speaker: Text
                if speaker != current_speaker:
                    if current_speaker is not None:
                         f_txt.write(f"\n\n{current_speaker}: {' '.join(current_block)}")
                    current_speaker = speaker
                    current_block = [text]
                else:
                    current_block.append(text)
                
                # Timestamp format: [start - end] Speaker: Text
                f_ts.write(f"[{start_fmt} - {end_fmt}] {speaker}: {text}\n")
                
            # Write last block
            if current_speaker is not None:
                f_txt.write(f"\n\n{current_speaker}: {' '.join(current_block)}")
                
        print(f"Completed {input_file.name}")

def convert_to_wav_to_path(input_path: Path, output_path: Path) -> Path:
    """Convert audio/video file to WAV using ffmpeg to specific output path."""
    try:
        subprocess.run(
            ["ffmpeg", "-i", str(input_path), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(output_path), "-y"],
            check=True,
            capture_output=True
        )
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
        raise

if __name__ == "__main__":
    main()
