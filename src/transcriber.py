from pathlib import Path
import sys
import logging
import torch
import whisper
import soundfile as sf
import numpy as np
import datetime
import subprocess  # Usado para conversão M4A -> WAV
import language_tool_python
import torchaudio  # Normalização de áudio
import librosa
import gc


class Transcriber:
    def __init__(self, model_size="medium", language="pt", use_gpu=True,
                output_dir="transcriptions", max_audio_length=300, min_duration=2,
                enable_denoise=True, enable_correction=True):
        self.model_size = model_size
        self.language = language
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.output_dir = Path(output_dir)
        self.max_audio_length = max_audio_length
        self.min_duration = min_duration
        self.enable_denoise = enable_denoise
        self.enable_correction = enable_correction  # 🔹 Novo parâmetro opcional

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True
        )
        self.logger = logging.getLogger(__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        if self.enable_correction:
            import language_tool_python  # 🔹 Importa apenas se necessário
            self.grammar_tool = language_tool_python.LanguageTool("pt-BR")
    
    def log_step(self, message):
        """Registra logs de execução."""
        self.logger.info(message)
        sys.stdout.flush()

    def load_model(self):
        """Carrega o modelo Whisper na GPU se disponível."""
        self.log_step("Step 1/5: Loading Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(self.model_size, device=device)
        self.log_step("✅ Whisper model loaded successfully.")

    def transcribe_audio(self, audio_path):
        """
        Full pipeline for transcribing a single audio file.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            self.log_step(f"❌ File not found: {audio_path}")
            return

        # 🔄 Converte para WAV se necessário
        audio_path = self._convert_audio(audio_path)
        if audio_path is None:
            self.log_step(f"❌ Audio conversion failed. Skipping {audio_path.name}.")
            return
        
        self.log_step(f"🎙️ Processing: {audio_path.name}")

        # Step 1: Split audio into segments (only if needed)
        segments = self._split_audio(audio_path)
        if not segments:
            self.log_step(f"⚠️ No valid segments found for {audio_path.name}. Skipping.")
            return

        transcribed_text = []
        accumulated_time = 0

        # Step 2: Transcribe each segment sequentially
        for segment in segments:
            self.log_step(f"📝 Transcribing segment: {segment.name}")
            formatted_text, duration = self._transcribe_segment(segment, accumulated_time)
            if formatted_text:
                transcribed_text.append(formatted_text)
                accumulated_time += duration

            torch.cuda.empty_cache()
            gc.collect()

        if not transcribed_text:
            self.log_step(f"❌ Transcription failed for {audio_path.name}.")
            return

        final_text = "\n".join(transcribed_text)

        # Step 3: Apply grammar correction (only if enabled)
        if self.enable_correction:
            self.log_step(f"📖 Applying grammar correction for {audio_path.name}...")
            final_text = self.correct_text(final_text)

        # Step 4: Save transcription
        output_file = self.output_dir / f"{audio_path.stem}_transcription.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_text)

        self.log_step(f"✅ File saved: {output_file}")

        # Step 5: Remove temporary files
        self._cleanup_segments(segments)

    def _convert_m4a_to_wav(self, m4a_path):
        """Converte arquivos M4A para WAV usando FFmpeg."""
        wav_path = m4a_path.with_suffix(".wav")
        command = ["ffmpeg", "-i", str(m4a_path), "-ac", "1", "-ar", "16000", "-y", str(wav_path)]
        
        try:
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            self.log_step(f"✅ Conversion successful: {wav_path.name}")
            return wav_path
        except subprocess.CalledProcessError as e:
            self.log_step(f"❌ Error converting {m4a_path.name} to WAV: {e.stderr.decode()}")
            raise RuntimeError(f"FFmpeg failed to convert {m4a_path}")

    def _normalize_audio(self, audio_path):
        """Normaliza o volume do áudio para melhorar a transcrição."""
        self.log_step(f"🔹 Normalizing audio: {audio_path.name}")
        output_path = audio_path.with_suffix(".normalized.wav")

        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = torchaudio.functional.gain(waveform, gain_db=5)  # Aumenta o volume em 5dB

        torchaudio.save(str(output_path), waveform, sample_rate)
        return output_path

    def _transcribe_segment(self, segment_path, accumulated_time):
        """
        Transcreve um segmento de áudio com mais precisão.

        Args:
            segment_path (Path): Caminho do segmento de áudio.
            accumulated_time (float): Tempo acumulado para sincronizar a transcrição.

        Returns:
            tuple: Texto formatado e duração do segmento transcrito.
        """
        if not segment_path.exists():
            self.log_step(f"❌ Segmento não encontrado: {segment_path}")
            return "", 0

        try:
            # 🔹 Aumentando a precisão com temperatura variável
            result = self.model.transcribe(
                str(segment_path), language="pt", fp16=False, temperature=[0, 0.2, 0.5, 1.0]
            )
        except Exception as e:
            self.log_step(f"❌ Erro na transcrição de {segment_path.name}: {e}")
            return "", 0

        if "segments" not in result or not result["segments"]:
            return "", 0

        formatted_text = []
        for seg in result["segments"]:
            start_time = accumulated_time + seg["start"]
            text = seg["text"].strip()
            formatted_text.append(f"[{self._format_timestamp(start_time)}] {text}")

        segment_duration = result["segments"][-1]["end"] if result["segments"] else 0
        return "\n".join(formatted_text), segment_duration

    def _split_audio(self, audio_path):
        """
        Splits an audio file into smaller segments if it exceeds the max_audio_length.
        Uses FFmpeg for precise slicing.

        Args:
            audio_path (Path): Path to the audio file.

        Returns:
            List[Path]: List of paths to the generated audio segments.
        """
        self.log_step(f"🔹 Checking audio duration - {audio_path.name}")
        
        duration = librosa.get_duration(path=str(audio_path))
        
        if duration <= self.max_audio_length:
            self.log_step(f"🎧 Audio duration is {duration:.2f}s (less than {self.max_audio_length} sec) - Skipping split.")
            return [audio_path]  # Retorna o áudio original se ele for curto

        self.log_step(f"✂️ Splitting audio: {audio_path.name} into segments of {self.max_audio_length} sec each")

        segments = []
        num_segments = int(duration // self.max_audio_length) + 1

        for i in range(num_segments):
            start_time = i * self.max_audio_length
            segment_path = self.output_dir / f"{audio_path.stem}_seg_{i}.wav"

            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(audio_path),
                "-ss", str(start_time),
                "-t", str(self.max_audio_length),
                "-ac", "1", "-ar", "16000", "-y",
                str(segment_path)
            ]

            try:
                subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                self.log_step(f"✅ Created segment: {segment_path.name}")
                segments.append(segment_path)
            except subprocess.CalledProcessError as e:
                self.log_step(f"❌ FFmpeg error splitting audio: {e}")

        self.log_step(f"🔹 Total segments created for {audio_path.name}: {len(segments)}")
        return segments

    def _denoise_audio(self, audio_path):
        """
        Aplica redução de ruído no áudio antes da transcrição.

        Args:
            audio_path (Path): Caminho do arquivo de áudio.

        Returns:
            tuple: Áudio processado e taxa de amostragem.
        """
        self.log_step(f"🔹 Aplicando redução de ruído em: {audio_path.name}")

        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # 🔹 Filtro passa-alta para remover ruídos de baixa frequência (exemplo: vento, vibração)
            waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate, cutoff_freq=150)

            # 🔹 Filtro passa-baixa para remover ruídos de alta frequência (exemplo: chiados, apitos)
            waveform = torchaudio.functional.lowpass_biquad(waveform, sample_rate, cutoff_freq=7500)

            self.log_step("✅ Redução de ruído concluída.")
            return waveform, sample_rate

        except Exception as e:
            self.log_step(f"❌ Erro na redução de ruído: {e}")
            return torchaudio.load(audio_path)  # Retorna o áudio original em caso de erro

    def correct_text(self, text):
        """Corrige erros gramaticais no texto transcrito."""
        return self.grammar_tool.correct(text)

    def _cleanup_segments(self, segments):
        """
        Remove os arquivos temporários criados durante a transcrição.

        Args:
            segments (list): Lista de caminhos dos arquivos de áudio segmentados a serem deletados.
        """
        if not segments:
            self.log_step("🗑️ Nenhum segmento para limpar.")
            return

        self.log_step(f"🗑️ Deletando {len(segments)} segmentos temporários...")

        for segment in segments:
            try:
                if segment.exists():
                    segment.unlink()
                    self.log_step(f"✅ Deletado: {segment.name}")
            except Exception as e:
                self.log_step(f"❌ Erro ao deletar {segment.name}: {e}")

        self.log_step("✅ Limpeza de arquivos temporários concluída.")

    def _convert_audio(self, input_path):
        """
        Converts an audio file to WAV format if it's not already in WAV format.
        
        Args:
            input_path (Path): Path to the original audio file.
        
        Returns:
            Path: Path to the converted WAV file.
        """
        input_path = Path(input_path)
        if input_path.suffix.lower() == ".wav":
            return input_path  # Se já for WAV, não precisa converter
        
        self.log_step(f"🔄 Converting {input_path.suffix.upper()} to WAV: {input_path.name}")
        
        output_path = input_path.with_suffix(".wav")  # Define o novo arquivo WAV
        command = [
            "ffmpeg", "-i", str(input_path),
            "-ac", "1", "-ar", "16000", "-y", str(output_path)  # Mono e 16kHz (padrão do Whisper)
        ]

        try:
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            self.log_step(f"✅ Conversion successful: {output_path.name}")
            return output_path
        except subprocess.CalledProcessError as e:
            self.log_step(f"❌ Error converting audio: {e}")
            return None

    def _format_timestamp(self, seconds):
        """Converte segundos para formato HH:MM:SS."""
        return str(datetime.timedelta(seconds=int(seconds)))
