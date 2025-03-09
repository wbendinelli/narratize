import os
import torch
from TTS.api import TTS

class PodcastGeneratorXTTS:
    def __init__(self, language="pt", output_dir="output_podcast",
                 speed=1.0, emotion="neutral", speaker_wav=None):
        """
        Inicializa o gerador de podcasts usando XTTS v2.

        Parâmetros:
        - language (str): Idioma ("pt" para português ou "en" para inglês).
        - output_dir (str): Diretório para salvar os arquivos gerados.
        - speed (float): Velocidade da fala (1.0 = normal, >1.0 = mais rápido, <1.0 = mais lento).
        - emotion (str): Emoção da voz ("neutral", "happy", "sad", "angry").
        - speaker_wav (str): Caminho para um arquivo de áudio para clonagem de voz (opcional).
        """
        self.language = language
        self.output_dir = output_dir
        self.speed = speed
        self.emotion = emotion
        self.speaker_wav = speaker_wav  # Arquivo de referência para imitar uma voz específica
        os.makedirs(output_dir, exist_ok=True)

        # Inicializa o modelo XTTS v2
        print("🚀 Carregando modelo XTTS v2...")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

        # Define o uso de GPU se disponível
        if torch.cuda.is_available():
            self.tts.to("cuda")
            print("✅ Rodando na GPU")
        else:
            print("⚠️ Rodando no CPU (pode ser mais lento).")

    def generate_podcast(self, script_text, output_filename="podcast_output.mp3"):
        """
        Gera um arquivo de áudio a partir de um texto de roteiro.

        Parâmetros:
        - script_text (str): O texto do roteiro do podcast.
        - output_filename (str): Nome do arquivo de saída.
        """
        print(f"🎙️ Gerando áudio para o podcast ({self.language})...")
        output_path = os.path.join(self.output_dir, output_filename)

        # Gera o áudio a partir do texto com parâmetros personalizados
        self.tts.tts_to_file(
            text=script_text,
            language=self.language,
            file_path=output_path,
            speed=self.speed,
            emotion=self.emotion,
            speaker_wav=self.speaker_wav  # Usa a clonagem de voz se um arquivo for fornecido
        )

        print(f"✅ Podcast gerado com sucesso: {output_path}")
        return output_path

    def generate_podcast_from_file(self, script_file, output_filename="podcast_output.mp3"):
        """
        Gera um arquivo de áudio a partir de um roteiro salvo em arquivo.

        Parâmetros:
        - script_file (str): Caminho do arquivo de roteiro.
        - output_filename (str): Nome do arquivo de saída.
        """
        if not os.path.exists(script_file):
            raise FileNotFoundError(f"❌ Arquivo de roteiro não encontrado: {script_file}")

        # Ler o roteiro do podcast
        with open(script_file, "r", encoding="utf-8") as f:
            script_text = f.read().strip()

        return self.generate_podcast(script_text, output_filename)