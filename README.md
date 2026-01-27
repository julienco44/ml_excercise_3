# Next-Word Prediction: Preprocessing Pipeline

This repository contains a robust, industry-standard preprocessing pipeline for Deep Learning-based Next-Word Prediction (Language Modeling). It is designed to prepare text data for training RNNs (LSTMs, GRUs) or Transformers.

## 🚀 Pipeline Overview

The pipeline transforms raw text into efficient PyTorch datasets through the following rigorous steps:

### 1. Text Cleaning & Normalization (`text_cleaner.py`)
- **Punctuation Handling:** Crucially, sentence-ending punctuation (`.`, `?`, `!`) is **preserved** and treated as separate tokens. This allows the model to learn sentence boundaries and structural breaks.
- **Noise Removal:** Removes special characters, URLs, and emails while keeping legitimate words and structure.
- **Normalization:** Converts text to lowercase and (optionally) removes accents using `unidecode`.

### 2. Tokenization & Special Tokens (`pipeline.py`)
- **Word-Level Tokenization:** Splits text into individual words.
- **EOS Injection:** Appends an `<EOS>` (End of Sentence) token to the end of every document. This teaches the model when a text is finished, which is essential for generation tasks.

### 3. Leakage-Free Splitting (`pipeline.py`)
- **Document-Level Split:** Data is split into Train/Validation/Test sets *before* chunking.
- **Why?** Splitting after chunking causes "Data Leakage" where parts of the same article appear in both Train and Test, inflating accuracy scores. Our approach ensures the Test set contains strictly unseen documents.

### 4. Sliding Window Chunking (`pipeline.py`)
- **No Truncation:** Instead of cutting off texts at 50 words, we slice long documents into multiple contiguous chunks (e.g., words 0-50, 50-100, etc.).
- **Benefit:** Utilizes 100% of the available training data.

### 5. Efficient Dataset Creation (`dataset.py`)
- **Shifted Tensors:** The dataset produces `(Input, Target)` pairs in $O(N)$ time.
    - **Input:** Sequence of length $N$ (e.g., `[A, B, C]`)
    - **Target:** The same sequence shifted by 1 (e.g., `[B, C, D]`)
- **Performance:** This allows the model to predict the next word for *every* position in the sequence simultaneously during a single forward pass.

## 🛠 Usage

### Prerequisites
Install dependencies:
```bash
pip install -r preprocessing_pipeline/requirements.txt
```

### Running the Pipeline
Run the main script to download the **20Newsgroups** dataset and process it:

```bash
python preprocessing_pipeline/main.py --dataset 20newsgroups --max-seq-length 50
```

### Output
Processed data is saved to `processed_data/`:
- `vocab.pkl`: Serialized `Vocabulary` object (word <-> index mapping).
- `train_dataset.pkl`: Training chunks (PyTorch Dataset).
- `val_dataset.pkl`: Validation chunks.
- `test_dataset.pkl`: Test chunks.

## 📈 Advanced Features
- **Semantic Evaluation:** The `Vocabulary` class supports loading pretrained embeddings (like GloVe) to evaluate predictions based on semantic similarity (synonyms) rather than just exact matches.

```python
# Example: Loading embeddings for evaluation
vocab.load_embeddings("path/to/glove.txt")
```
