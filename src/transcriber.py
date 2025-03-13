from pathlib import Path
import sys
import logging
import torch
import whisper
import soundfile as sf
import numpy as np
import librosa
import gc
import language_tool_python
import torchaudio
import subprocess
from huggingface_hub import login
from pyannote.audio.pipelines import SpeakerDiarization
import datetime


class Transcriber:
    """
    Processes and transcribes audio files using OpenAI's Whisper model.
    Supports segmentation, speaker diarization, and grammar correction.
    """

    def __init__(self, model_size="small", language="pt", use_gpu=True,
                 output_dir="transcriptions", segment_duration=60,
                 diarization_model="pyannote/speaker-diarization-3.1", hf_auth_token=None,
                 clustering_threshold=0.85, min_silence_duration=1.5):
        """
        Initializes the transcriber with specified parameters.
        """
        self.model_size = model_size
        self.language = language
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.output_dir = Path(output_dir)
        self.segment_duration = segment_duration
        self.hf_auth_token = hf_auth_token
        self.clustering_threshold = clustering_threshold
        self.min_silence_duration = min_silence_duration

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True
        )
        self.logger = logging.getLogger(__name__)

        self.model = None
        self.tool = language_tool_python.LanguageTool("pt-BR")

        # Load diarization model
        self._load_diarization_model(diarization_model)

    def _load_diarization_model(self, diarization_model):
        """Loads the speaker diarization model with explicit authentication."""
        self.log_step("Loading speaker diarization model...")

        if self.hf_auth_token:
            login(self.hf_auth_token)  # Ensure authentication before loading the model

        try:
            device = torch.device("cuda" if self.use_gpu else "cpu")
            self.diarization_pipeline = SpeakerDiarization.from_pretrained(
                diarization_model, use_auth_token=self.hf_auth_token
            ).to(device)
            self.log_step("Speaker diarization model loaded successfully.")
        except Exception as e:
            self.diarization_pipeline = None
            self.log_step(f"Error loading speaker diarization model: {e}")

    def _perform_diarization(self, audio_path):
        """Performs speaker diarization with optimized parameters."""
        if not self.diarization_pipeline:
            self.log_step("Diarization model not loaded. Skipping speaker identification.")
            return {}

        self.log_step(f"Performing diarization on {audio_path.name}...")

        waveform, sample_rate = torchaudio.load(audio_path)

        params = {
            "segmentation": {"min_duration_off": self.min_silence_duration},
            "clustering": {"threshold": self.clustering_threshold}
        }

        self.diarization_pipeline.instantiate(params)
        diarization = self.diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})

        speaker_timestamps = {}
        speaker_mapping = {}
        speaker_counter = 1

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start, end = turn.start, turn.end

            if speaker not in speaker_mapping:
                speaker_mapping[speaker] = f"Speaker {speaker_counter}"
                speaker_counter += 1

            speaker_name = speaker_mapping[speaker]
            speaker_timestamps.setdefault(speaker_name, []).append((start, end))

        self.log_step(f"Diarization completed. Identified {len(speaker_mapping)} speakers.")
        return speaker_timestamps

    def log_step(self, message):
        """Logs messages with INFO level."""
        self.logger.info(message)
        sys.stdout.flush()

    def load_model(self):
        """Loads the Whisper model into memory."""
        self.log_step("Loading Whisper model...")
        device = "cuda" if self.use_gpu else "cpu"

        try:
            self.model = whisper.load_model(self.model_size, device=device)
            self.log_step("Whisper model successfully loaded.")
        except Exception as e:
            self.log_step(f"Error loading Whisper model: {e}")
            raise

    def transcribe_audio(self, audio_path):
        """Processes an audio file, applying diarization, segmentation, transcription, and grammar correction."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            self.log_step(f"File not found: {audio_path}")
            return

        self.log_step(f"Processing: {audio_path.name}")

        speaker_timestamps = self._perform_diarization(audio_path)
        segments = self._split_audio(audio_path)

        if not segments:
            self.log_step(f"No valid segments found for {audio_path.name}. Skipping.")
            return

        transcribed_text = []
        accumulated_time = 0

        for idx, segment in enumerate(segments):
            self.log_step(f"Transcribing segment {idx + 1}/{len(segments)}: {segment.name}")
            formatted_text, duration = self._transcribe_segment(segment, accumulated_time, speaker_timestamps)

            if formatted_text:
                transcribed_text.append(formatted_text)
                accumulated_time += duration

            torch.cuda.empty_cache()
            gc.collect()

        if not transcribed_text:
            self.log_step(f"Transcription failed for {audio_path.name}.")
            return

        final_text = "\n".join(transcribed_text)
        self.log_step(f"Applying grammar correction for {audio_path.name}...")
        final_text = self.correct_text(final_text)

        output_file = self.output_dir / f"{audio_path.stem}_transcription.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_text)

        self.log_step(f"File saved: {output_file}")
        self._cleanup_segments(segments)

    def _split_audio(self, audio_path):
        """Splits an audio file into smaller segments using ffmpeg."""
        self.log_step(f"Splitting audio: {audio_path.name}")

        duration = librosa.get_duration(path=audio_path)
        segments = []

        if duration <= self.segment_duration:
            return [audio_path]

        for i in range(0, int(duration), self.segment_duration):
            segment_path = self.output_dir / f"{audio_path.stem}_seg_{i // self.segment_duration}.wav"

            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(audio_path), "-ss", str(i), "-t", str(self.segment_duration),
                "-ac", "1", "-ar", "16000", "-y", str(segment_path)
            ]

            try:
                subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                segments.append(segment_path)
            except subprocess.CalledProcessError as e:
                self.log_step(f"FFmpeg error: {e.stderr.decode()}")

        return segments

    def _transcribe_segment(self, segment_path, accumulated_time, speaker_timestamps):
        """Transcribes an individual segment and assigns speakers when applicable."""
        if not segment_path.exists():
            self.log_step(f"Skipping missing segment: {segment_path}")
            return "", 0

        try:
            audio = whisper.load_audio(str(segment_path))
            result = self.model.transcribe(audio, language=self.language)
        except Exception as e:
            self.log_step(f"Error transcribing segment {segment_path.name}: {e}")
            return "", 0

        if "segments" not in result or not result["segments"]:
            return "", 0

        formatted_text = []
        for seg in result["segments"]:
            start_time = accumulated_time + seg["start"]
            text = seg["text"].strip()
            speaker = self._assign_speaker(start_time, speaker_timestamps)

            formatted_text.append(f"[{self._format_timestamp(start_time)}] [{speaker}]: {text}")

        segment_duration = result["segments"][-1]["end"] if result["segments"] else 0
        return "\n".join(formatted_text), segment_duration

    def _assign_speaker(self, timestamp, speaker_timestamps):
        """Assigns a speaker label to a given timestamp."""
        for speaker, times in speaker_timestamps.items():
            for start, end in times:
                if start <= timestamp <= end:
                    return speaker
        return "Unknown Speaker"

    def _format_timestamp(self, seconds):
        """Converts seconds into HH:MM:SS format."""
        return str(datetime.timedelta(seconds=int(seconds)))

    def correct_text(self, text):
        """Applies grammar correction using LanguageTool."""
        return self.tool.correct(text)

    def _cleanup_segments(self, segments):
        """Deletes temporary audio segments after processing."""
        for segment in segments:
            if segment.exists():
                segment.unlink()
                self.log_step(f"Deleted segment: {segment.name}")

#Exemple Usage
from pathlib import Path
import os
import sys

# Adjust the module path to ensure correct import
sys.path.append("/content/drive/MyDrive/narratize/src/")

# Attempt to import Transcriber
try:
    from transcriber import Transcriber
except ModuleNotFoundError as e:
    print(f"Error importing Transcriber: {e}")
    print("Check if 'transcriber.py' is located in '/content/drive/MyDrive/narratize/src/'.")
    sys.exit(1)  # Exit the script if the module is not found

# Define input and output directories
INPUT_FOLDER = Path("/content/drive/MyDrive/")  # Directory containing audio files
OUTPUT_FOLDER = Path("/content/drive/MyDrive/narratize/data/transcriptions")  # Directory for saving transcriptions

# Set and authenticate the Hugging Face Token
HF_AUTH_TOKEN = "YOUR_TOKEN_HERE"  # Replace with your actual token
os.environ["HUGGINGFACE_TOKEN"] = HF_AUTH_TOKEN  # Make the token globally accessible

# Ensure the output directory exists
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Initialize the transcriber
transcriber = Transcriber(
    model_size="small",
    language="pt",
    use_gpu=True,
    output_dir=OUTPUT_FOLDER,
    segment_duration=60,  # Duration of audio segments
    clustering_threshold=0.85,  # Threshold for speaker differentiation
    min_silence_duration=1.5,  # Minimum silence duration to detect speaker change
    hf_auth_token=HF_AUTH_TOKEN  # Hugging Face token for diarization
)

# Load the Whisper model
transcriber.load_model()

# List audio files in the directory
audio_files = list(INPUT_FOLDER.glob("*.mp3")) + list(INPUT_FOLDER.glob("*.wav"))

# Check if there are any audio files to process
if not audio_files:
    print("No audio files found in the specified folder.")
    sys.exit(0)  # Exit without error if no files are found

print(f"Found {len(audio_files)} audio files. Starting transcription...")

# Process each audio file
for idx, audio_file in enumerate(audio_files, start=1):
    print(f"\nProcessing {idx}/{len(audio_files)}: {audio_file.name}")
    transcriber.transcribe_audio(audio_file)

print("\nAll files have been processed. Transcriptions are saved in:", OUTPUT_FOLDER)
