import streamlit as st
import torch
from transcriber import Transcriber
import os
from pathlib import Path

# ✅ Page Configuration
st.set_page_config(
    page_title="🎙️ AI Audio Transcription",
    layout="centered"
)

# ✅ Custom Styling for a Professional Look
st.markdown("""
    <style>
    .main { text-align: center; }
    .css-18e3th9 { padding-top: 2rem; }
    .stButton>button, .stDownloadButton>button { 
        width: 100%; 
        font-size: 16px; 
        border-radius: 8px;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #45a049;
    }
    .stTextArea textarea {
        font-size: 14px;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Header
st.title("🎙️ AI-Powered Audio Transcription")
st.markdown("""
    Transform your audio into **high-quality text** with cutting-edge AI technology.  
    **Supported formats:** WAV, MP3, M4A.  
""")

# 🔹 Model Selection (Limited to "small" for Streamlit compatibility)
st.subheader("⚙️ Settings")
model_size = st.selectbox(
    "Choose a Whisper Model:",
    ["tiny", "base", "small"],
    index=2
)

# 🔹 Language Selection
language = st.selectbox("Select Language:", ["English", "Portuguese", "Spanish"])
lang_code = {"English": "en", "Portuguese": "pt", "Spanish": "es"}[language]

# 🔹 GPU Option (If Available)
use_gpu = st.checkbox("Use GPU (if available)", value=torch.cuda.is_available())

# 🔹 File Upload
st.subheader("📂 Upload Your Audio File")
uploaded_file = st.file_uploader("Drag and drop or browse for an audio file:", type=["wav", "mp3", "m4a"])

if uploaded_file:
    # ✅ Audio Player
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

    # ✅ Transcription Button
    if st.button("🔍 Start Transcription"):
        with st.spinner("⏳ Processing... This may take a few seconds."):
            # ✅ Save the uploaded file temporarily
            temp_audio_path = Path(f"temp_audio.{uploaded_file.name.split('.')[-1]}")
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.read())

            try:
                # ✅ Initialize Transcriber
                transcriber = Transcriber(model_size=model_size, language=lang_code, use_gpu=use_gpu)
                transcriber.load_model()

                # ✅ Perform Transcription
                transcription = transcriber.transcribe_audio(str(temp_audio_path))

                # ✅ Display Transcription
                st.success("✅ Transcription Completed!")
                st.subheader("📜 Transcribed Text:")
                st.text_area("Output:", transcription, height=300)

                # 🔥 Download Button
                st.download_button(
                    "⬇️ Download Transcription",
                    transcription,
                    file_name="transcription.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"❌ An error occurred during transcription: {str(e)}")

            finally:
                # 🗑️ Securely Remove Temporary Files
                if temp_audio_path.exists():
                    os.remove(temp_audio_path)
