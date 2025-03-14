import streamlit as st
import torch
import whisper
import torchaudio
import os
import numpy as np
from pydub import AudioSegment
from pathlib import Path

# ✅ Configuração inicial do Streamlit
st.set_page_config(page_title="🎙️ Transcrição de Áudio", layout="centered")

st.title("🎙️ Transcrição de Áudio com IA")
st.write("Faça upload de um arquivo de áudio para transcrição! (Formatos suportados: WAV, MP3, M4A)")

# 🔹 Escolha do modelo Whisper
model_size = st.selectbox("Selecione o modelo Whisper:", ["tiny", "base", "small", "medium", "large"])

# 🔹 Escolha do idioma
language = st.selectbox("Escolha o idioma:", ["Português", "Inglês", "Espanhol"])
lang_code = {"Português": "pt", "Inglês": "en", "Espanhol": "es"}[language]

# 🔹 Opção de GPU (se disponível)
use_gpu = st.checkbox("Usar GPU (se disponível)", value=torch.cuda.is_available())

# 🔹 Upload do arquivo de áudio (suporta MP3, M4A, WAV)
uploaded_file = st.file_uploader("Faça upload do arquivo de áudio", type=["wav", "mp3", "m4a"])


def load_audio(input_audio, format):
    """
    Carrega o áudio de entrada sem conversão.
    Utiliza torchaudio para WAV e pydub para MP3 e M4A.
    Retorna o waveform (Tensor) e a taxa de amostragem.
    """
    if format == "wav":
        waveform, sample_rate = torchaudio.load(input_audio)
    else:
        audio = AudioSegment.from_file(input_audio, format=format)
        sample_rate = audio.frame_rate
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        waveform = torch.tensor(samples).unsqueeze(0) / 32768.0  # Normalizar

    return waveform, sample_rate


if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("🔍 Transcrever"):
        with st.spinner("⏳ Transcrevendo... Aguarde..."):
            # ✅ Salvar o arquivo temporariamente
            temp_audio_path = Path(f"temp_audio.{uploaded_file.name.split('.')[-1]}")
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.read())

            # ✅ Carregar o áudio diretamente sem conversão
            file_format = temp_audio_path.suffix[1:]  # Pega extensão sem o ponto
            waveform, sample_rate = load_audio(str(temp_audio_path), file_format)

            # ✅ Carregar o modelo Whisper
            device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            model = whisper.load_model(model_size, device=device)

            # ✅ Transcrição do áudio
            result = model.transcribe(str(temp_audio_path), language=lang_code)

            # ✅ Exibir transcrição
            st.subheader("📜 Transcrição:")
            st.text_area("Resultado:", result["text"], height=300)

            # 🔥 Salvar a transcrição
            output_file = Path("transcription.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result["text"])

            st.success("✅ Transcrição concluída e salva!")

            # 🔥 Botão para baixar a transcrição
            with open(output_file, "rb") as f:
                st.download_button("⬇️ Baixar Transcrição", f, file_name="transcription.txt", mime="text/plain")

            # 🗑️ Remover arquivos temporários
            os.remove(temp_audio_path)
