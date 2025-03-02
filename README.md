# 📚 Narratize: AI-Driven Audio-to-Text Structuring Framework

## **1. Introduction**
Narratize is a modular, AI-powered pipeline designed to process and structure audio content into textual formats optimized for research, publication, and podcasting. The framework leverages state-of-the-art natural language processing (NLP) techniques to perform high-fidelity **transcription, summarization, and structuring**, enabling automated generation of structured documents, book chapters, and podcast-ready scripts.

This project is particularly useful in scenarios where **long-form content needs to be efficiently transformed into structured, coherent, and domain-specific narratives**.

---

## **2. Technical Overview**
Narratize is built using a hybrid AI approach, integrating:
- **Speech-to-Text (ASR)**: OpenAI’s Whisper model for high-accuracy, multi-language transcription.
- **Text Summarization**: Transformer-based models such as BART, T5, and Pegasus for abstractive and extractive summarization.
- **Structured Text Generation**: Large Language Models (LLMs) such as GPT-4 for chapter-style content development.
- **Multi-Format Export**: Markdown, Notion API, and structured text for various downstream applications.

### **2.1 Workflow Architecture**
```mermaid
graph TD;
    A[Audio/Video Input] -->|Whisper| B[Raw Transcription]
    B -->|Summarization (BART/T5/Pegasus)| C[Structured Summary]
    C -->|LLM Expansion (GPT-4)| D[Book Chapter]
    C -->|Formatted Dialogue| E[Podcast Script]
    D -->|Markdown/Notion Export| F[Book Draft]
    E -->|Audio Generation| G[Podcast]
```

### **2.2 Core Modules**
- **`transcriber.py`**: ASR module for speech-to-text conversion.
- **`structurer.py`**: Text post-processing, summarization, and structuring.
- **`chapter_creator.py`**: Advanced structuring using LLMs for book-style narratives.
- **`exporter.py`**: Integration with external systems (Markdown, Notion, Google Docs).

---

## **3. Installation & Setup**

### **3.1 Requirements**
```bash
pip install -r requirements.txt
```

If running on **Google Colab**, mount the Google Drive for persistent storage:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Then navigate to the project directory:
```bash
cd /content/drive/MyDrive/narratize/
```

---

## **4. Usage Instructions**

### **4.1 Transcribe Audio (Whisper ASR)**
```python
from src.transcriber import Transcriber

transcriber = Transcriber(model_size="medium", language="en", verbose=True)
transcriber.transcribe_audio("data/input/sample.mp3", "data/transcriptions/transcribed_sample.txt")
```

### **4.2 Summarization & Structuring**
```python
from src.structurer import TextStructurer

structurer = TextStructurer(models=["facebook/bart-large-cnn", "t5-small"], verbose=True)
summary = structurer.summarize_text("data/transcriptions/transcribed_sample.txt", model="facebook/bart-large-cnn")
```

### **4.3 Long-Form Chapter Generation (LLM Augmentation)**
```python
from src.chapter_creator import ChapterCreator

chapter_creator = ChapterCreator(model="gpt-4", api_key="YOUR_OPENAI_KEY")
book_chapter = chapter_creator.generate_chapter(summary, book_genre="Technical Writing")
```

### **4.4 Exporting to Notion API**
```python
from src.exporter import NotionExporter

notion = NotionExporter(api_key="YOUR_NOTION_API_KEY")
notion.upload_to_notion(book_chapter, page_title="New Chapter")
```

---

## **5. Evaluation Metrics & Benchmarking**
Narratize incorporates various **quantitative evaluation metrics** to assess transcription accuracy, summarization fidelity, and structured output coherence:

| **Metric**              | **Description** |
|------------------------|--------------------------------------------------------------------|
| **WER (Word Error Rate)** | Measures ASR transcription accuracy. Lower is better.          |
| **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** | Assesses summarization overlap. |
| **BERTScore**          | Semantic similarity score between generated and reference texts. |
| **Compression Ratio**  | Evaluates content reduction efficiency without loss of meaning.  |

To benchmark performance, execute:
```python
from evaluation import evaluate_model

evaluate_model("data/transcriptions/transcribed_sample.txt", "data/structured_text/structured_sample.txt")
```

---

## **6. Applications & Research Potential**
- **Automated Audiobook and Podcast Production**: Streamline the conversion of spoken content into structured books and formatted dialogues.
- **Lecture & Meeting Summarization**: Enhance accessibility of recorded academic and business discussions.
- **NLP & AI Research**: Utilize structured AI-generated text for further **fine-tuning models**.
- **Legal & Compliance Documentation**: Transform recorded testimonies into structured, reviewable documentation.
- **Digital Humanities & Archival Research**: Automate transcription and structuring of historical recordings.

---

## **7. Contribution Guidelines**
We encourage contributions from the research and developer community. To contribute:
1. **Fork the repository**.
2. **Create a new feature branch** (`feature-xyz`).
3. **Follow the PEP8/PEP257 coding style**.
4. **Submit a PR with detailed documentation and test cases**.

Ensure all new features pass existing unit tests:
```bash
pytest tests/
```

---

## **8. License & Citation**
This project is released under the **MIT License**.
For academic use, please cite:
```latex
@misc{narratize2024,
  title={Narratize: AI-Driven Audio Structuring Framework},
  author={William Bendinelli et al.},
  year={2024},
  howpublished={\url{https://github.com/wbendinelli/narratize}}
}
```

---

## **9. Roadmap & Future Work**
- 📌 **Fine-tune ASR models for specialized domains**
- 📌 **Implement multi-modal summarization pipelines**
- 📌 **Develop interactive UI for text generation refinement**
- 📌 **Enhance LLMs to support domain-specific chapter structuring**

---

**🚀 Narratize: A Research-Grade AI Framework for Structured Audio-to-Text Processing**
