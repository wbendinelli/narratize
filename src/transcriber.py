import whisper
import os
import datetime
import json
import logging
from pathlib import Path
import time
import sys  # Required to ensure logs display correctly in Colab

class Transcriber:
    """
    A class to handle audio transcription using the Whisper model.

    This class provides functionality for transcribing audio files, saving outputs in multiple formats,
    and managing transcription logs for better tracking.
    """

    def __init__(self, model_size="medium", language=None, temperature=0.0, 
                 no_speech_threshold=0.3, logprob_threshold=-1.0, compression_ratio_threshold=2.5, 
                 verbose=False, output_dir="/content/drive/MyDrive/narratize/data/transcriptions"):
        """
        Initializes the Whisper transcription model with customizable parameters.

        Parameters:
            model_size (str): Whisper model size. Options: "tiny", "base", "small", "medium", "large".
            language (str, optional): Preferred language for transcription (None for auto-detection).
            temperature (float): Decoding temperature (0 = deterministic, higher = more variation).
            no_speech_threshold (float): Minimum probability threshold to detect speech.
            logprob_threshold (float): Minimum log probability for valid transcription.
            compression_ratio_threshold (float): Maximum allowed compression ratio for a valid transcription.
            verbose (bool): If True, enables detailed logging.
            output_dir (str): Directory where transcriptions will be stored.
        """
        self.model_size = model_size
        self.language = language
        self.temperature = temperature
        self.no_speech_threshold = no_speech_threshold
        self.logprob_threshold = logprob_threshold
        self.compression_ratio_threshold = compression_ratio_threshold
        self.verbose = verbose
        self.output_dir = Path(output_dir)

        # Configure logging to ensure real-time display in Colab
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True
        )
        self.logger = logging.getLogger(__name__)

        # Ensure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._log_step("[1/6] Initializing Transcriber...")
        self._log_step(f"Model: {model_size}, Language: {language or 'Auto'}, Temperature: {temperature}")

        # Load Whisper model
        self._log_step("[2/6] Loading Whisper model...")
        self.model = whisper.load_model(model_size)
        self._log_step("Whisper model loaded successfully.")

    def _clear_previous_outputs(self):
        """
        Deletes all previous transcription files in the output directory before running a new transcription.

        Ensures that previous outputs do not interfere with new transcriptions.
        """
        self._log_step(f"[3/6] Deleting previous transcription files in: {self.output_dir}")
        for file in self.output_dir.glob("*"):
            try:
                file.unlink()
                self._log_step(f"Deleted: {file}")
            except Exception as e:
                self.logger.warning(f"Could not delete {file}: {e}")
        self._log_step("Cleanup completed.")

    def transcribe_audio(self, audio_path, output_file=None, save_txt=True, save_srt=True, save_json=True):
        """
        Transcribes an audio file using Whisper and saves the transcript in various formats.

        Parameters:
            audio_path (str): Path to the input audio file.
            output_file (str, optional): Custom filename for the transcription (without extension).
            save_txt (bool): If True, saves the transcription as a .txt file.
            save_srt (bool): If True, saves subtitles as a .srt file.
            save_json (bool): If True, saves the full Whisper output as a .json file.

        Returns:
            dict: Whisper transcription result.
        """
        # Convert to Path object for better path handling
        audio_path = Path(audio_path)

        # Validate if the audio file exists
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._log_step(f"[4/6] Preparing transcription for: {audio_path.name}")

        # Remove previous transcription files before starting a new one
        self._clear_previous_outputs()

        # Generate default filename if not provided
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = audio_path.stem
            output_file = self.output_dir / f"{base_name}_{timestamp}"

        output_file = Path(output_file)

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Start transcription
        self._log_step("[5/6] Running Whisper model for transcription...")
        transcribe_params = {
            "language": self.language,
            "temperature": self.temperature,
            "no_speech_threshold": self.no_speech_threshold,
            "logprob_threshold": self.logprob_threshold,
            "compression_ratio_threshold": self.compression_ratio_threshold
        }
        transcribe_params = {k: v for k, v in transcribe_params.items() if v is not None}

        result = self.model.transcribe(str(audio_path), **transcribe_params)

        # Ensure the transcription contains text
        if not result["text"].strip():
            raise ValueError("No transcription output detected. Check the audio file or parameters.")

        self._log_step("Transcription completed successfully.")

        # Save transcription in different formats
        self._log_step("[6/6] Saving transcription outputs...")
        if save_txt:
            txt_file = output_file.with_suffix(".txt")
            self._save_text(result["text"], txt_file)

        if save_srt:
            srt_file = output_file.with_suffix(".srt")
            self._save_srt(result, srt_file)

        if save_json:
            json_file = output_file.with_suffix(".json")
            self._save_json(result, json_file)

        # Force Google Drive sync
        self._sync_google_drive()

        self._log_step("All files saved successfully.")
   
        return result
    
    def _save_text(self, text, file_path):
        """Saves the transcribed text as a .txt file."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        self._log_step(f"Saved transcription (TXT): {file_path}")

    def _save_srt(self, result, file_path):
        """Saves the transcription as an .srt subtitle file."""
        with open(file_path, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                start = self.format_timestamp(segment["start"])
                end = self.format_timestamp(segment["end"])
                text = segment["text"]
                f.write(f"{segment['id'] + 1}\n{start} --> {end}\n{text}\n\n")
        self._log_step(f"Saved subtitles (SRT): {file_path}")

    def _save_json(self, result, file_path):
        """Saves the full Whisper output as a JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        self._log_step(f"Saved transcription metadata (JSON): {file_path}")

    def _sync_google_drive(self):
        """Forces synchronization with Google Drive to ensure files are saved properly."""
        self._log_step("Forcing Google Drive sync...")
        time.sleep(5)
        self._log_step("Google Drive sync completed.")

    @staticmethod
    def format_timestamp(seconds):
        """Formats a timestamp to HH:MM:SS,MS format for SRT files."""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

    def _log_step(self, message):
        """Helper function to log steps for visibility in Colab."""
        print(message)
        sys.stdout.flush()
        self.logger.info(message)