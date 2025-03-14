import streamlit as st
import torch
import whisper
import torchaudio
import os
from pathlib import Path

# ✅ Configuração inicial do Streamlit
st.set_page_config(page_title="🎙️ Transcrição de Áudio", layout="centered")

st.title("🎙️ Transcrição de Áudio com IA")
st.write("Faça upload de um arquivo de áudio em formato WAV para transcrição!")

# 🔹 Escolha do modelo Whisper
model_size = st.selectbox("Selecione o modelo Whisper:", ["tiny", "base", "small", "medium", "large"])

# 🔹 Escolha do idioma
language = st.selectbox("Escolha o idioma:", ["Português (pt)", "Inglês (en)", "Espanhol (es)"])
lang_code = {"Português (pt)": "pt", "Inglês (en)": "en", "Espanhol (es)": "es"}[language]

# 🔹 Opção de GPU (se disponível)
use_gpu = st.checkbox("Usar GPU (se disponível)", value=torch.cuda.is_available())

# 🔹 Upload do arquivo de áudio (Apenas WAV)
uploaded_file = st.file_uploader("Faça upload do arquivo de áudio (somente WAV)", type=["wav"])


if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("🔍 Transcrever"):
        with st.spinner("⏳ Transcrevendo... Aguarde..."):
            # ✅ Salvar o arquivo temporariamente
            temp_audio_path = Path("temp_audio.wav")
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.read())

            # ✅ Garantir que torchaudio use `sox_io`
            torchaudio.set_audio_backend("sox_io")

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
