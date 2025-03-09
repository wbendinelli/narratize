from pathlib import Path
import sys
import logging
import torch
import whisper
import soundfile as sf
import numpy as np
import datetime
import librosa
import gc
import language_tool_python

class Transcriber:
    """
    A class for processing and transcribing audio files efficiently using OpenAI's Whisper model.
    It supports splitting large audio files, speaker recognition, and grammar correction.
    """

    def __init__(self, model_size="small", language="pt", use_gpu=True, output_dir="transcriptions",
                 segment_duration=60, silence_threshold=1.5):
        """
        Initializes the transcriber with specified parameters.

        Args:
            model_size (str): Whisper model size to use ('tiny', 'small', 'medium', 'large').
            language (str): Language for transcription.
            use_gpu (bool): Whether to use GPU acceleration.
            output_dir (str): Directory for saving transcriptions.
            segment_duration (int): Maximum duration of each audio segment in seconds.
            silence_threshold (float): Minimum silence gap to detect speaker change.
        """
        self.model_size = model_size
        self.language = language
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.output_dir = Path(output_dir)
        self.segment_duration = segment_duration  
        self.silence_threshold = silence_threshold  

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True
        )
        self.logger = logging.getLogger(__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tool = language_tool_python.LanguageTool("pt-BR")

    def log_step(self, message):
        """
        Logs messages with INFO level.
        
        Args:
            message (str): Log message.
        """
        self.logger.info(message)
        sys.stdout.flush()

    def load_model(self):
        """ Loads the Whisper model into memory. """
        self.log_step("Loading Whisper model...")
        device = "cuda" if self.use_gpu else "cpu"

        try:
            self.model = whisper.load_model(self.model_size, device=device)
            self.log_step("Whisper model successfully loaded.")
        except Exception as e:
            self.log_step(f"Error loading Whisper model: {e}")
            raise

    def transcribe_audio(self, audio_path):
        """
        Full pipeline for transcribing a single audio file.

        Args:
            audio_path (str or Path): Path to the audio file.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            self.log_step(f"File not found: {audio_path}")
            return

        self.log_step(f"Processing: {audio_path.name}")

        # Step 1: Split audio into segments
        segments = self._split_audio(audio_path)
        if not segments:
            self.log_step(f"No valid segments found for {audio_path.name}. Skipping.")
            return

        transcribed_text = []
        accumulated_time = 0

        # Step 2: Transcribe each segment sequentially
        for segment in segments:
            self.log_step(f"Transcribing segment: {segment.name}... Running in background.")
            formatted_text, duration = self._transcribe_segment(segment, accumulated_time)
            if formatted_text:
                transcribed_text.append(formatted_text)
                accumulated_time += duration

            torch.cuda.empty_cache()
            gc.collect()

        if not transcribed_text:
            self.log_step(f"Transcription failed for {audio_path.name}.")
            return

        final_text = "\n".join(transcribed_text)

        # Step 3: Apply grammar correction
        self.log_step(f"Applying grammar correction for {audio_path.name}...")
        final_text = self.correct_text(final_text)

        # Step 4: Save transcription
        output_file = self.output_dir / f"{audio_path.stem}_transcription.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_text)

        self.log_step(f"File saved: {output_file}")

        # Step 5: Remove temporary files
        self._cleanup_segments(segments)

    def _split_audio(self, audio_path):
        """
        Splits an audio file into smaller segments.

        Args:
            audio_path (Path): Path to the audio file.

        Returns:
            list: List of segmented audio file paths.
        """
        self.log_step(f"Splitting audio: {audio_path.name}")

        try:
            audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            self.log_step(f"Error loading audio file {audio_path.name}: {e}")
            return []

        duration = librosa.get_duration(y=audio, sr=sample_rate)

        if duration <= self.segment_duration:
            return [audio_path]

        segments = []
        num_segments = int(np.ceil(duration / self.segment_duration))
        overlap = 0.5  # 500ms overlap to avoid cut-off issues

        for i in range(num_segments):
            start_time = max(0, i * self.segment_duration - overlap)
            end_time = min(duration, (i + 1) * self.segment_duration)

            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) / sample_rate < 2:
                continue  

            segment_path = self.output_dir / f"{audio_path.stem}.part{i}.wav"
            sf.write(segment_path, segment_audio, sample_rate)
            segments.append(segment_path)
            self.log_step(f"Created segment {i+1}/{num_segments}: {segment_path.name}")

        return segments

    def _cleanup_segments(self, segments):
        """
        Deletes temporary audio segments after processing.

        Args:
            segments (list): List of segment file paths.
        """
        for segment in segments:
            if Path(segment).exists():
                Path(segment).unlink()
                self.log_step(f"Deleted segment: {segment.name}")

    def _transcribe_segment(self, segment_path, accumulated_time):
        """
        Transcribes an audio segment.

        Args:
            segment_path (Path): Path to the audio segment.
            accumulated_time (float): Time offset for accurate timestamps.

        Returns:
            tuple: (Transcribed text, segment duration)
        """
        if not segment_path.exists():
            self.log_step(f"Skipping missing segment: {segment_path}")
            return "", 0

        try:
            audio = whisper.load_audio(str(segment_path))
            if audio.shape[0] == 0:
                self.log_step(f"Skipping empty segment: {segment_path.name}")
                return "", 0

            result = self.model.transcribe(audio, language=self.language)
        except Exception as e:
            self.log_step(f"Error transcribing with Whisper: {e}")
            return "", 0

        if "segments" not in result or not result["segments"]:
            return "", 0

        formatted_text = []
        last_timestamp = 0
        speaker_count = 1

        for seg in result["segments"]:
            start_time = accumulated_time + seg["start"]
            text = seg["text"].strip()

            if seg["start"] - last_timestamp > self.silence_threshold:
                speaker_count += 1

            formatted_text.append(f"[{self._format_timestamp(start_time)}] [Speaker {speaker_count}] {text}")
            last_timestamp = seg["end"]

        segment_duration = result["segments"][-1]["end"] if result["segments"] else 0
        return "\n".join(formatted_text), segment_duration

    def _format_timestamp(self, seconds):
        """
        Converts time in seconds to HH:MM:SS format.

        Args:
            seconds (int): Time in seconds.

        Returns:
            str: Formatted timestamp in HH:MM:SS.
        """
        return str(datetime.timedelta(seconds=int(seconds)))

    def correct_text(self, text):
        """
        Applies grammar correction using LanguageTool.

        Args:
            text (str): Transcribed text.

        Returns:
            str: Corrected text with improved grammar.
        """
        return self.tool.correct(text)

# Exempla Usage

if __name__ == "__main__":
    # Directory containing audio files
    audio_dir = Path("/content/drive/MyDrive/audio_files")
    output_dir = Path("/content/drive/MyDrive/narratize/data/transcriptions")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of all audio files (MP3 and WAV)
    audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))

    if not audio_files:
        print("No audio files found in the directory.")
        sys.exit()

    # Initialize the transcriber
    transcriber = Transcriber(model_size="small", language="pt", use_gpu=True, output_dir=output_dir)

    # Load the Whisper model once
    transcriber.load_model()

    # Process each file sequentially
    for audio_file in audio_files:
        transcriber.transcribe_audio(audio_file)