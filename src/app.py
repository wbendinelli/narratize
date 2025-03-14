import streamlit as st
import torch
import whisper
import soundfile as sf
import numpy as np
import os
from pathlib import Path

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    import torchaudio
    LIBROSA_AVAILABLE = False


# ✅ Configuração inicial do Streamlit
st.set_page_config(page_title="🎙️ Transcrição de Áudio", layout="centered")

st.title("🎙️ Transcrição de Áudio com IA")
st.write("Faça upload de um arquivo de áudio e obtenha a transcrição!")

# 🔹 Escolha do modelo Whisper
model_size = st.selectbox("Selecione o modelo Whisper:", ["tiny", "base", "small", "medium", "large"])

# 🔹 Escolha do idioma
language = st.selectbox("Escolha o idioma:", ["Português (pt)", "Inglês (en)", "Espanhol (es)"])
lang_code = {"Português (pt)": "pt", "Inglês (en)": "en", "Espanhol (es)": "es"}[language]

# 🔹 Opção de GPU (se disponível)
use_gpu = st.checkbox("Usar GPU (se disponível)", value=torch.cuda.is_available())

# 🔹 Upload do arquivo de áudio
uploaded_file = st.file_uploader("Faça upload do arquivo de áudio", type=["wav", "mp3", "m4a"])


def load_audio(input_audio):
    """Carrega qualquer formato de áudio usando librosa (se disponível) ou torchaudio como fallback."""
    if LIBROSA_AVAILABLE:
        audio_data, sample_rate = librosa.load(input_audio, sr=16000, mono=True)
    else:
        audio_data, sample_rate = torchaudio.load(input_audio)
        audio_data = audio_data.mean(dim=0).numpy()  # Converte para mono

    output_wav = Path("temp_audio.wav")
    sf.write(output_wav, audio_data, sample_rate)
    return output_wav


if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("🔍 Transcrever"):
        with st.spinner("⏳ Transcrevendo... Aguarde..."):
            # ✅ Salvar o arquivo temporariamente
            temp_audio_path = Path(f"temp_audio.{uploaded_file.name.split('.')[-1]}")
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.read())

            # ✅ Carregar o áudio usando librosa ou torchaudio
            temp_audio_path = load_audio(str(temp_audio_path))

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
