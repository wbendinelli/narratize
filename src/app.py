import streamlit as st
import torch
from transcriber import Transcriber
import os
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

# 🔹 Upload do arquivo de áudio
uploaded_file = st.file_uploader("Faça upload do arquivo de áudio", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("🔍 Transcrever"):
        with st.spinner("⏳ Transcrevendo... Aguarde..."):
            # ✅ Salvar o arquivo temporariamente
            temp_audio_path = Path(f"temp_audio.{uploaded_file.name.split('.')[-1]}")
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.read())

            # ✅ Criar instância do transcriber
            transcriber = Transcriber(model_size=model_size, language=lang_code, use_gpu=use_gpu)
            transcriber.load_model()

            # ✅ Fazer a transcrição
            transcription = transcriber.transcribe_audio(str(temp_audio_path))

            # ✅ Exibir transcrição
            st.subheader("📜 Transcrição:")
            st.text_area("Resultado:", transcription, height=300)

            # 🔥 Botão para baixar a transcrição
            st.download_button("⬇️ Baixar Transcrição", transcription, file_name="transcription.txt", mime="text/plain")

            # 🗑️ Remover arquivos temporários
            os.remove(temp_audio_path)
