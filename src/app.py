import streamlit as st
from transcriber import Transcriber

st.title("🎙️ Transcrição de Áudio com IA")

uploaded_file = st.file_uploader("Faça o upload de um arquivo de áudio", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("🔍 Transcrever"):
        transcriber = Transcriber(model_size="large-v2", output_dir="transcriptions")
        transcriber.load_model()
        transcription = transcriber.transcribe_audio(uploaded_file)
        st.text_area("📜 Transcrição:", transcription, height=300)
