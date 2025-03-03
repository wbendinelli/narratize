import os
import json
import logging
from pathlib import Path
import torch
from transformers import pipeline, AutoTokenizer

class TextStructurer:
    def __init__(self, models=None, 
                 max_summary_length=300, min_summary_length=100, 
                 stride=256, verbose=True, device=None):
        """
        Initializes the TextStructurer with multiple summarization models.

        Parameters:
            models (list): List of Hugging Face models for summarization.
            max_summary_length (int): Maximum length of summaries.
            min_summary_length (int): Minimum length of summaries.
            stride (int): Overlapping tokens to retain context.
            verbose (bool): Whether to print logs.
            device (str): Device to use ('cpu', 'cuda', or None for auto-selection).
        """
        self.verbose = verbose
        self.max_summary_length = max_summary_length
        self.min_summary_length = min_summary_length
        self.stride = stride
        self.models = models or ["facebook/bart-large-cnn", "google/pegasus-cnn_dailymail", "t5-small"]

        # Configure logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - [%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        # Automatically select device if not provided
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.logger.info(f"STEP 1/3: Using device: {self.device}")

        # Load summarization models
        self.summarizers = {}
        self.tokenizers = {}
        for model in self.models:
            self.logger.info(f"  - Loading summarization model: {model}...")
            try:
                self.summarizers[model] = pipeline("summarization", model=model, device=0 if self.device == "cuda" else -1)
                self.tokenizers[model] = AutoTokenizer.from_pretrained(model)
                self.logger.info(f"  - Model {model} successfully loaded.")
            except Exception as e:
                self.logger.error(f"Failed to load model {model}: {str(e)}")

    def summarize_text(self, text):
        """
        Summarizes the input text using multiple models.

        Parameters:
            text (str): Input text for summarization.

        Returns:
            dict: Dictionary with model names as keys and their corresponding summaries.
        """
        self.logger.info("STEP 2/3: Summarizing text with multiple models...")

        summaries = {}
        for model, summarizer in self.summarizers.items():
            tokenizer = self.tokenizers[model]
            token_limit = min(1024, self.max_summary_length * 2)  # Ajuste do tamanho máximo permitido

            tokens = tokenizer(text, truncation=False, return_tensors="pt")
            num_tokens = tokens.input_ids.shape[1]

            if num_tokens > token_limit:
                self.logger.info(f"  - Text exceeds model token limit ({token_limit} tokens), processing in chunks...")
                chunks = self._split_into_chunks(text, tokenizer, token_limit)
            else:
                chunks = [text]

            model_summaries = []
            for i, chunk in enumerate(chunks):
                self.logger.info(f"    - Processing chunk {i+1}/{len(chunks)}")
                summary = summarizer(chunk, 
                                     max_length=self.max_summary_length, 
                                     min_length=self.min_summary_length, 
                                     do_sample=False)
                model_summaries.append(summary[0]["summary_text"])

            summaries[model] = " ".join(model_summaries)

        return summaries

    def _split_into_chunks(self, text, tokenizer, token_limit):
        """
        Splits a long text into smaller chunks ensuring each chunk is within model token limits.

        Parameters:
            text (str): The input text.
            tokenizer: Tokenizer of the selected model.
            token_limit (int): Maximum token limit per chunk.

        Returns:
            list: List of text chunks.
        """
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = min(start + token_limit, len(words))
            chunk = " ".join(words[start:end])

            # Ensure the chunk doesn't exceed token limit
            while len(tokenizer(chunk)["input_ids"]) > token_limit and end > start:
                end -= 1
                chunk = " ".join(words[start:end])

            chunks.append(chunk)
            start = end  # Move forward

        return chunks

    def process_transcription(self, input_file, output_dir=None, output_filename="structured_output"):
        """
        Processes a transcribed file and summarizes its content with multiple models.

        Parameters:
            input_file (str): Path to the transcribed text file.
            output_dir (str, optional): Directory to save the structured output.
            output_filename (str): Custom output filename (without extension).

        Returns:
            dict: Dictionary with summaries from each model.
        """
        self.logger.info("STEP 3/3: Processing transcription file...")

        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Transcribed file not found: {input_file}")

        with open(input_path, "r", encoding="utf-8") as f:
            transcribed_text = f.read()

        summaries = self.summarize_text(transcribed_text)

        output_dir = Path(output_dir or input_path.parent)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_files = {}
        for model, summary in summaries.items():
            model_name = model.replace("/", "_")  # Safe filename
            output_path = output_dir / f"{output_filename}_{model_name}.txt"

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(summary)

            self.logger.info(f"  - Summarized text saved at: {output_path}")
            summary_files[model] = output_path

        return summary_files


# ---------------------- EXAMPLE USAGE ----------------------
if __name__ == "__main__":
    structurer = TextStructurer(
        models=["facebook/bart-large-cnn", "google/pegasus-cnn_dailymail", "t5-small"],  # Multiple models
        verbose=True,
        max_summary_length=300,
        min_summary_length=100,
        stride=256,  # Overlap to maintain context
        device="auto"  # Automatically select CPU/GPU
    )

    input_file = "/content/drive/MyDrive/narratize/data/transcriptions/transcribed_sample.txt"
    output_dir = "/content/drive/MyDrive/narratize/data/structured"
    output_filename = "structured_output"

    summary_results = structurer.process_transcription(
        input_file=input_file,
        output_dir=output_dir,
        output_filename=output_filename
    )

    print("\n--- Summarized Outputs ---\n")
    for model, file_path in summary_results.items():
        print(f"Model: {model} -> Summary saved at: {file_path}")