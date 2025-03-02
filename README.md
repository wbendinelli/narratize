# 📚 Narratize: Transform Audio into Books & Podcasts

Narratize is an **AI-powered pipeline** that **transcribes**, **summarizes**, and **structures** audio content into **books or podcasts**.

---

## **🚀 Features**
✅ **Automatic Transcription** using Whisper  
✅ **Summarization with BART, T5, and Pegasus**  
✅ **Notion Integration** for structured storage  
✅ **Export to Markdown, Notion, and PDF**  
✅ **Support for Podcast Script Generation**  

---

## **📚 Project Structure**
```
narratize/
│── data/
│   ├── input/             # Raw audio/video files
│   ├── transcriptions/    # Whisper transcriptions
│   ├── structured_text/   # Summarized and structured content
│   ├── notion_sync/       # Notion-ready exports
│   ├── book_output/       # Final book-ready chapters
│   ├── podcast_output/    # Final podcast-ready scripts
│── src/                   # Core processing scripts
│── utils/                 # Helper scripts
│── configs/               # Configuration settings
│── notebooks/             # Jupyter Notebooks for testing
│── tests/                 # Unit tests
│── docs/                  # Documentation
│── README.md              # Project documentation
```

---

## **🔄 Workflow**
```mermaid
graph TD;
    A[Audio/Video Input] -->|Whisper| B[Raw Transcription]
    B -->|Summarization (BART/T5/Pegasus)| C[Structured Summary]
    C -->|LLM Expansion (GPT-4)| D[Book Chapter]
    C -->|Formatted Dialogue| E[Podcast Script]
    D -->|Markdown/Notion Export| F[Book Draft]
    E -->|Audio Generation| G[Podcast]
```

1️⃣ **Transcribe Audio (Whisper)**  
2️⃣ **Summarize & Structure Text (BART, T5, Pegasus)**  
3️⃣ **Export to Notion / Markdown**  
4️⃣ **Format for Book or Podcast Production**  

---

## **🛠️ Installation**
```bash
pip install -r requirements.txt
```

If running on Google Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Then navigate to the project directory:
```bash
cd /content/drive/MyDrive/narratize/
```

---

## **🚀 How to Run**
### **1️⃣ Transcribe Audio (Whisper)**
```python
from src.transcriber import Transcriber

transcriber = Transcriber(model_size="medium", language="en", verbose=True)
transcriber.transcribe_audio("data/input/sample.mp3", "data/transcriptions/transcribed_sample.txt")
```

### **2️⃣ Summarize & Structure Text**
```python
from src.structurer import TextStructurer

structurer = TextStructurer(models=["facebook/bart-large-cnn", "t5-small"], verbose=True)
summary = structurer.summarize_text("data/transcriptions/transcribed_sample.txt", model="facebook/bart-large-cnn")
```

### **3️⃣ Expand Summary into a Book Chapter**
```python
from src.chapter_creator import ChapterCreator

chapter_creator = ChapterCreator(model="gpt-4", api_key="YOUR_OPENAI_KEY")
book_chapter = chapter_creator.generate_chapter(summary, book_genre="Science Fiction")
```

### **4️⃣ Export to Notion**
```python
from src.exporter import NotionExporter

notion = NotionExporter(api_key="YOUR_NOTION_API_KEY")
notion.upload_to_notion(book_chapter, page_title="New Chapter")
```

---

## **📚 Output Examples**
### **🔹 Summarized Content**
```text
In a world where artificial intelligence governs human society, a young hacker uncovers a secret that could change everything...
```

### **🔹 Expanded Book Chapter**
```markdown
# Chapter 1: The Hidden Code

In the sprawling megacity of Neo-Tokyo, the streets hummed with the quiet murmur of drones and self-driving taxis. Among them, a hooded figure slipped unnoticed through the shadows...

"Are you sure about this?" asked Alex, his fingers hovering over the keyboard. The encrypted message glowed on his screen...
```

### **🔹 Structured Podcast Script**
```text
Alice: "Did you know that AI can now write entire books?"
Bob: "That's fascinating! But what does it mean for authors?"
Alice: "Well, it could revolutionize storytelling..."
```

---

## **⚙️ Configuration (Optional)**
You can customize settings in `configs/settings.yaml`:
```yaml
whisper_model: "medium"
language: "en"
max_summary_length: 350
min_summary_length: 120
export_format: "notion"
```

---

## **📈 Possible Applications**
✅ **Automated Audiobook Creation**  
✅ **Content Repurposing for Blogs, Newsletters, and Podcasts**  
✅ **Summarizing Lectures and Meetings**  
✅ **Generating Training Material for AI & NLP Research**  
✅ **Academic Research - Converting Transcribed Interviews into Papers**  

---

## **👨‍💻 Contributing**
We welcome contributions!  
To contribute, fork the repo, make your changes, and submit a PR.  
Make sure to follow our coding guidelines in `.github/CONTRIBUTING.md`.

---

## **📃 License**
This project is licensed under the **MIT License**.

---

**🚀 Ready to transform your audio into books & podcasts? Let's get started with Narratize!**

