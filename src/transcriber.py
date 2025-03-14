import os
import sys
import logging
import torch
import whisper
import soundfile as sf
import numpy as np
import datetime
from pathlib import Path
from pydub import AudioSegment  # 🔥 Alternativa ao ffmpeg
import torchaudio  # 🔥 Alternativa para carregar áudios sem conversão

class Transcriber:
    def __init__(self, model_size="small", language="pt", use_gpu=True,
                 output_dir="transcriptions", max_audio_length=300, min_duration=2):
        self.model_size = model_size
        self.language = language
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.output_dir = Path(output_dir)
        self.max_audio_length = max_audio_length
        self.min_duration = min_duration

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True
        )
        self.logger = logging.getLogger(__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None

    def log_step(self, message):
        self.logger.info(message)
        sys.stdout.flush()

    def load_model(self):
        self.log_step("Step 1/5: Carregando o modelo Whisper...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(self.model_size, device=device)
        self.log_step("Modelo Whisper carregado com sucesso.")

    def transcribe_audio(self, audio_path):
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")

        # 🔥 Converter para WAV (caso seja MP3 ou M4A)
        wav_path = self._convert_to_wav(audio_path)

        self.log_step(f"Step 2/5: Verificando e dividindo áudio, se necessário - {wav_path.name}")
        segments = self._split_audio(wav_path)

        if not segments:
            self.log_step(f"Nenhum segmento encontrado para transcrição - {wav_path}")
            return

        output_file = self.output_dir / f"{audio_path.stem}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        transcribed_text = []
        accumulated_time = 0

        for idx, segment in enumerate(segments):
            self.log_step(f"Step 3/5: Transcrevendo segmento {idx+1}/{len(segments)} - {segment.name}")
            formatted_text, segment_duration = self._transcribe_segment(segment, accumulated_time)
            if formatted_text:
                transcribed_text.append(formatted_text)
                accumulated_time += segment_duration

        final_text = "\n".join(transcribed_text)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_text)
        self.log_step(f"✔ Arquivo TXT salvo: {output_file}")

        self._cleanup_segments(segments)

    def _convert_to_wav(self, audio_path):
        """Converte MP3/M4A para WAV se necessário."""
        if audio_path.suffix.lower() in [".wav"]:
            return audio_path  # Já está no formato correto

        output_wav = audio_path.with_suffix(".wav")
        self.log_step(f"🔄 Convertendo {audio_path.suffix} → WAV: {output_wav}")
        audio = AudioSegment.from_file(audio_path)
        audio.export(output_wav, format="wav")
        return output_wav

    def _split_audio(self, audio_path):
        """Divide o áudio se for maior que `max_audio_length`."""
        audio, sample_rate = sf.read(audio_path)
        duration = len(audio) / sample_rate
        if duration <= self.max_audio_length:
            return [audio_path]

        segments = []
        num_segments = int(np.ceil(duration / self.max_audio_length))
        for i in range(num_segments):
            start_sample = int(i * self.max_audio_length * sample_rate)
            end_sample = min(int((i+1) * self.max_audio_length * sample_rate), len(audio))
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) / sample_rate < self.min_duration:
                continue

            segment_path = audio_path.with_suffix(f".part{i}.wav")
            sf.write(segment_path, segment_audio, sample_rate)
            segments.append(segment_path)

        return segments

    def _transcribe_segment(self, segment_path, accumulated_time):
        """Executa a transcrição do segmento de áudio."""
        result = self.model.transcribe(str(segment_path), language=self.language, fp16=False)
        formatted_text = []
        for segment in result['segments']:
            start_time = accumulated_time + segment['start']
            end_time = accumulated_time + segment['end']
            text = segment['text'].strip()
            formatted_text.append(f"[{self._format_timestamp(start_time)}] Speaker: {text}")

        segment_duration = result['segments'][-1]['end']
        return "\n".join(formatted_text), segment_duration

    def _cleanup_segments(self, segments):
        """Remove arquivos temporários."""
        for segment in segments:
            if segment.exists():
                segment.unlink()

    def _format_timestamp(self, seconds):
        """Formata timestamps no estilo HH:MM:SS."""
        return str(datetime.timedelta(seconds=int(seconds)))

# ✅ Exemplo de uso:
if __name__ == "__main__":
    audio_file = "/content/drive/MyDrive/audio_teste_1.mp3"
    transcriber = Transcriber(output_dir="/content/drive/MyDrive/narratize/data/transcriptions")
    transcriber.load_model()
    transcriber.transcribe_audio(audio_file)
