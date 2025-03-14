import os
import streamlit as st
import torch
import whisper
from pathlib import Path

# ✅ Instala o FFmpeg automaticamente se não estiver disponível
if not os.system("ffmpeg -version") == 0:
    st.warning("FFmpeg não encontrado! Instalando agora...")
    os.system("apt-get install -y ffmpeg")

# 🔹 Configuração inicial do Streamlit
st.set_page_config(page_title="🎙️ Transcrição de Áudio", layout="centered")

st.title("🎙️ Transcrição de Áudio com IA")
st.write("Faça upload de um arquivo de áudio e obtenha a transcrição!")

# ✅ Escolha do modelo Whisper
model_size = st.selectbox("Selecione o modelo Whisper:", ["tiny", "base", "small", "medium", "large"])

# ✅ Escolha do idioma
language = st.selectbox("Escolha o idioma:", ["Português (pt)", "Inglês (en)", "Espanhol (es)"])
lang_code = {"Português (pt)": "pt", "Inglês (en)": "en", "Espanhol (es)": "es"}[language]

# ✅ Opção de GPU (se disponível)
use_gpu = st.checkbox("Usar GPU (se disponível)", value=torch.cuda.is_available())

# 🔹 Upload do arquivo de áudio
uploaded_file = st.file_uploader("Faça upload do arquivo de áudio", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")
    
    if st.button("🔍 Transcrever"):
        with st.spinner("⏳ Transcrevendo... Aguarde..."):
            # ✅ Salva o arquivo temporariamente
            temp_audio_path = Path(f"temp_audio.{uploaded_file.name.split('.')[-1]}")
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.read())

            # ✅ Carrega o modelo Whisper
            device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            model = whisper.load_model(model_size, device=device)

            # ✅ Faz a transcrição
            result = model.transcribe(str(temp_audio_path), language=lang_code)

            # ✅ Exibe a transcrição na tela
            st.subheader("📜 Transcrição:")
            st.text_area("Resultado:", result["text"], height=300)

            # 🔥 Salvar a transcrição em um arquivo
            output_file = Path("transcription.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result["text"])

            st.success("✅ Transcrição concluída e salva como 'transcription.txt'.")

            # 🔥 Botão para baixar o arquivo de transcrição
            with open(output_file, "rb") as f:
                st.download_button("⬇️ Baixar Transcrição", f, file_name="transcription.txt", mime="text/plain")

            # 🗑️ Remove o arquivo temporário
            os.remove(temp_audio_path)
