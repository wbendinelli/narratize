import streamlit as st
import torch
import os
from pathlib import Path
from transcriber import Transcriber
from structurer import TextStructurer
import openai

# ✅ Page Configuration
st.set_page_config(page_title="🎙️ AI Audio Transcription & Structuring", layout="centered")

# ✅ Custom Styling for a Professional Look
st.markdown("""
    <style>
    .main { text-align: center; }
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
st.title("🎙️ AI-Powered Audio Transcription & Structuring")
st.markdown("Upload an audio file to generate a structured report using **Whisper AI & GPT**. 🚀")

# 🔹 Model Selection (Limited to "small" for Streamlit compatibility)
st.subheader("⚙️ Settings")
model_size = st.selectbox("Choose a Whisper Model:", ["tiny", "base", "small"], index=2)

# 🔹 Language Selection
language = st.selectbox("Select Language:", ["English", "Portuguese", "Spanish"])
lang_code = {"English": "en", "Portuguese": "pt", "Spanish": "es"}[language]

# 🔹 GPU Option (If Available)
use_gpu = st.checkbox("Use GPU (if available)", value=torch.cuda.is_available())

# 🔹 OpenAI API Key
st.subheader("🔑 OpenAI API Key")
api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# 🔹 File Upload
st.subheader("📂 Upload Your Audio File")
uploaded_file = st.file_uploader("Drag and drop or browse for an audio file:", type=["wav", "mp3", "m4a"])

if uploaded_file:
    # ✅ Audio Player
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

    # ✅ Log Display Section
    log_placeholder = st.empty()  # Placeholder for real-time logs

    # ✅ Transcription Button
    if st.button("🔍 Start Transcription"):
        with st.spinner("⏳ Processing... Please wait."):

            # ✅ Save uploaded file temporarily
            temp_audio_path = Path(f"temp_audio.{uploaded_file.name.split('.')[-1]}")
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.read())

            try:
                # ✅ Initialize Transcriber
                transcriber = Transcriber(model_size=model_size, language=lang_code, use_gpu=use_gpu)
                transcriber.load_model()

                # ✅ Process Transcription with Live Logs
                transcription_logs = []
                for log_message in transcriber.transcribe_with_logs(str(temp_audio_path)):
                    transcription_logs.append(log_message)
                    log_placeholder.text_area("📜 Transcription Log:", "\n".join(transcription_logs), height=300)

                # ✅ Display Final Transcription
                transcription = "\n".join(transcription_logs)
                st.success("✅ Transcription Completed!")
                st.subheader("📜 Transcribed Text:")
                st.text_area("Output:", transcription, height=300)

                # 🔥 Download Transcription
                st.download_button(
                    "⬇️ Download Transcription",
                    transcription,
                    file_name="transcription.txt",
                    mime="text/plain"
                )

                # ✅ **Structured Report Section**
                st.subheader("📄 Generate a Structured Report")

                # 🔹 User Custom Prompt
                user_prompt = st.text_area("Enter your custom prompt for GPT:", "")

                # ✅ Generate Report Button
                if st.button("📝 Generate Structured Report"):
                    if not api_key:
                        st.error("❌ Please enter a valid OpenAI API Key!")
                    else:
                        with st.spinner("⏳ Generating structured report..."):

                            # ✅ Set OpenAI API Key
                            openai.api_key = api_key

                            # ✅ Initialize Text Structurer
                            structurer = TextStructurer(model="gpt-3.5-turbo", temperature=0.5, max_tokens=800)

                            # ✅ Use Custom Prompt if Provided
                            if user_prompt.strip():
                                structured_text = structurer.summarize_text(transcription, user_prompt)
                            else:
                                structured_text = structurer.summarize_text(transcription, "podcast")

                            # ✅ Display Report
                            st.success("✅ Report Generated Successfully!")
                            st.subheader("📄 Structured Report")
                            st.text_area("Formatted Report:", structured_text, height=400)

                            # 🔥 Download Report
                            st.download_button(
                                "⬇️ Download Report",
                                structured_text,
                                file_name="structured_report.txt",
                                mime="text/plain"
                            )

            except Exception as e:
                st.error(f"❌ An error occurred during processing: {str(e)}")

            finally:
                # 🗑️ Securely Remove Temporary Files
                if temp_audio_path.exists():
                    os.remove(temp_audio_path)
