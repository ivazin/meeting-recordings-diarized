import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import datetime
import gc

# Configure Hugging Face cache to local models directory
# This must be done before importing libraries that use HF
root_dir = Path(__file__).resolve().parent
models_dir = root_dir / "models"
models_dir.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(models_dir / "huggingface")

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

def transcribe_gigaam(audio_path: str, download_root: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe audio using GigaAM-v3."""
    logger.info("Starting transcription with GigaAM-v3...")
    
    try:
        # Patch for gigaam compatibility with newer huggingface_hub
        import huggingface_hub.errors
        import huggingface_hub.utils
        if not hasattr(huggingface_hub.errors, "LocalEntryNotFoundError"):
            huggingface_hub.errors.LocalEntryNotFoundError = huggingface_hub.utils.EntryNotFoundError
            
        from gigaam import load_model
    except ImportError:
        logger.error("GigaAM library not found. Please install it: pip install gigaam")
        raise
        
    # Load model (assuming v3_ctc for strict ASR matching, or standard loading)
    # Using "v3_ctc" as it's a common target for robust ASR.
    # Check for MPS availability
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Loading GigaAM model on {device}...")
    
    try:
        model = load_model("v3_ctc", device=device, download_root=download_root)
    except Exception as e:
        logger.error(f"Failed to load GigaAM model: {e}")
        raise

    # PATCH: GigaAM vad_utils passes a directory to pyannote.audio.Model.from_pretrained,
    # which pyannote treats as a Repo ID and fails validation.
    # We patch it to point to the pytorch_model.bin file instead.
    try:
        import gigaam.vad_utils

        from pyannote.audio import Model
        from torch.torch_version import TorchVersion
        from pyannote.audio.core.task import Problem, Resolution, Specifications
        import os

        # Only patch if we haven't already (or just overwrite, it's safer to use a wrapper if we cared about state, but this is simple)
        # We need access to the original resolver, which is in the module.
        _original_resolve = gigaam.vad_utils.resolve_local_segmentation_path

        def _patched_load_segmentation_model(model_id: str) -> Model:
            local_path = _original_resolve(model_id=model_id)
            
            # Fix: append pytorch_model.bin if it's a directory
            if os.path.isdir(local_path):
                potential_bin = os.path.join(local_path, "pytorch_model.bin")
                if os.path.exists(potential_bin):
                    logger.debug(f"Patching GigaAM VAD path to: {potential_bin}")
                    local_path = potential_bin
            
            with torch.serialization.safe_globals(
                [TorchVersion, Problem, Specifications, Resolution]
            ):
                return Model.from_pretrained(local_path)
        
        gigaam.vad_utils.load_segmentation_model = _patched_load_segmentation_model
        logger.debug("Patched gigaam.vad_utils.load_segmentation_model for local path compatibility.")
        
    except Exception as e:
        logger.warning(f"Failed to patch gigaam.vad_utils: {e}")

    start_time = time.time()
    
    # transcribe_longform returns a list of segments/utterances with timestamps
    # Expected format of return: [{'text': '...', 'start': 0.0, 'end': 1.0}, ...] check docs?
    # Based on research: returns list of utterances.
    recognition = model.transcribe_longform(audio_path)
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"GigaAM Transcription complete. Took {duration:.2f} seconds.")
    
    # Convert to Whisper-compatible format
    full_text = []
    segments = []
    
    # transcribe_longform returns a list of segments/utterances with timestamps 
    # Format: [{'transcription': 'text', 'boundaries': (start, end)}, ...]
    
    for item in recognition:
        # access attributes safely, but we expect dict based on debug
        if isinstance(item, dict):
             text = item.get('transcription', "")
             boundaries = item.get('boundaries', (0.0, 0.0))
             start = boundaries[0]
             end = boundaries[1]
        else:
             # Fallback if structure changes (unlikely now)
             text = getattr(item, 'transcription', "") or getattr(item, 'text', "")
             start = getattr(item, 'start', 0.0)
             end = getattr(item, 'end', 0.0)
        
        full_text.append(text)
        segments.append({
            "start": start,
            "end": end,
            "text": text
        })

        
    return {
        "text": " ".join(full_text),
        "segments": segments
    }

def setup_diarization_pipeline(hf_token: str) -> Any:
    """Load and configure the pyannote.audio pipeline once."""
    logger.info("Loading pyannote.audio pipeline...")
    
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
    except Exception as e:
        logger.error(f"Failed to load pyannote pipeline: {e}")
        logger.error("Make sure you have accepted the user agreement on Hugging Face for pyannote/speaker-diarization-3.1")
        raise

    # Send pipeline to GPU (MPS) if available
    if torch.backends.mps.is_available():
         pipeline.to(torch.device("mps"))
         logger.info("Using MPS for diarization.")
    else:
         logger.info("Using CPU for diarization.")
         
    return pipeline

def diarize_audio(audio_path: str, pipeline: Any) -> Any:
    """Run diarization using the pre-loaded pipeline."""
    logger.info("Running diarization...")
    
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


import argparse

def main():
    parser = argparse.ArgumentParser(description="Transcribe and optionally diarize audio files.")
    parser.add_argument("--model", default="whisper", choices=["whisper", "gigaam"], help="Model to use: 'whisper' (default) or 'gigaam'.")
    parser.add_argument("--no-diarization", action="store_true", help="Skip speaker diarization step.")
    args = parser.parse_args()

    # 1. Configuration
    root_dir = Path(__file__).parent
    input_dir = root_dir / "input"
    output_base_dir = root_dir / "output"
    models_dir = root_dir / "models" # clean this up, use global models_dir? no, keep it local scope for clarity or reuse
    env_file = root_dir / ".env"
    
    # Ensure models directory exists
    models_dir.mkdir(exist_ok=True)
    
    input_dir = Path("/Users/user/Movies/2026-01-19")
    output_base_dir = input_dir / "transcripts"
    
    # Load environment variables
    load_env_file(env_file)
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable not found. Please output it in .env or export it.")
        logger.error("You need a token with access to pyannote/speaker-diarization-3.1")
        return

    # Initialize diarization pipeline if needed
    diarization_pipeline = None
    if not args.no_diarization:
        try:
            diarization_pipeline = setup_diarization_pipeline(hf_token)
        except Exception:
            return

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
        
    # Ensure output directory exists
    output_base_dir.mkdir(exist_ok=True)

    # List of extensions to process
    valid_extensions = {".mkv", ".mov", ".mp4", ".mp3", ".wav", ".m4a", ".ogg"}
    
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

        # Check if already processed
        check_file = file_output_dir / ("plain_transcript.txt" if args.no_diarization else "transcript_diarized.txt")
        if check_file.exists():
            logger.info(f"Skipping {input_file.name}: Output file {check_file.name} already exists.")
            continue

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
            if args.model == "gigaam":
                 transcription_result = transcribe_gigaam(str(audio_file_to_process), download_root=str(models_dir / "gigaam"))
            else:
                 transcription_result = transcribe_audio(str(audio_file_to_process))
        except Exception as e:
            logger.exception(f"Transcription failed for {input_file.name}")
            continue

        # --- WRITE PLAIN TRANSCRIPTS ALWAYS ---
        plain_txt = file_output_dir / "plain_transcript.txt"
        plain_ts = file_output_dir / "plain_transcript_with_timestamps.txt"
        
        logger.info(f"Writing plain transcripts to {plain_txt} and {plain_ts}")
        
        try:
            with open(plain_txt, "w") as f_ptxt, open(plain_ts, "w") as f_pts:
                # Write full text
                f_ptxt.write(transcription_result['text'].strip())
                
                # Write text with timestamps
                for seg in transcription_result['segments']:
                    start_fmt = format_timestamp(seg['start'])
                    end_fmt = format_timestamp(seg['end'])
                    text = seg['text'].strip()
                    f_pts.write(f"[{start_fmt} - {end_fmt}] {text}\n")
        except Exception as e:
             logger.error(f"Failed to write plain transcripts: {e}")

        if args.no_diarization:
            logger.info("Skipping diarization as requested.")
            print(f"Completed {input_file.name} (Transcription only)")
            continue


        # 4. Diarization
        try:
            diarization_result = diarize_audio(str(audio_file_to_process), diarization_pipeline)
        except Exception as e:
            logger.exception(f"Diarization failed for {input_file.name}")
            continue

        # 5. Merging
        final_segments = merge_results(transcription_result, diarization_result)
        
        # 6. Output (Diarized)
        output_txt = file_output_dir / "transcript_diarized.txt"
        output_ts = file_output_dir / "transcript_with_timestamps.txt"
        
        logger.info(f"Writing diarized results to {output_txt} and {output_ts}")
        
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

        # Cleanup intermediate WAV if it's distinct from input
        if audio_file_to_process != input_file and audio_file_to_process.exists():
             logger.info(f"Cleaning up converted WAV: {audio_file_to_process.name}")
             audio_file_to_process.unlink()

        # Cleanup to prevent memory leaks
        if 'transcription_result' in locals():
            del transcription_result
        if 'diarization_result' in locals():
            del diarization_result
        if 'final_segments' in locals():
            del final_segments
        
        # Force garbage collection
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

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
