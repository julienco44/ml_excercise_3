# Data loading utilities

import os
import pickle
import urllib.request
import zipfile
from typing import Tuple, List, Optional
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'preprocessing_pipeline'))


def load_processed_data(data_dir: str = "processed_data") -> Tuple:
    """Load vocab and train/val/test sequences from processed data dir."""
    print(f"Loading preprocessed data from {data_dir}...")

    # Load vocabulary
    from preprocessing_pipeline.vocabulary import Vocabulary
    vocab = Vocabulary()
    vocab.load(os.path.join(data_dir, "vocab.pkl"))

    # Load sequences
    with open(os.path.join(data_dir, "train_dataset.pkl"), "rb") as f:
        train_sequences = pickle.load(f)
    print(f"  Train sequences: {len(train_sequences)}")

    with open(os.path.join(data_dir, "val_dataset.pkl"), "rb") as f:
        val_sequences = pickle.load(f)
    print(f"  Val sequences: {len(val_sequences)}")

    with open(os.path.join(data_dir, "test_dataset.pkl"), "rb") as f:
        test_sequences = pickle.load(f)
    print(f"  Test sequences: {len(test_sequences)}")

    return vocab, train_sequences, val_sequences, test_sequences


def download_glove_embeddings(dim: int = 100, target_dir: str = "embeddings") -> Optional[str]:
    """Download GloVe embeddings if not already present."""
    os.makedirs(target_dir, exist_ok=True)

    glove_file = os.path.join(target_dir, f"glove.6B.{dim}d.txt")

    if os.path.exists(glove_file):
        print(f"GloVe embeddings found at {glove_file}")
        return glove_file

    print(f"Downloading GloVe embeddings ({dim}d)...")
    zip_path = os.path.join(target_dir, "glove.6B.zip")

    url = "http://nlp.stanford.edu/data/glove.6B.zip"

    try:
        # Download with progress
        def _progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rDownloading: {percent}%", end='', flush=True)

        urllib.request.urlretrieve(url, zip_path, _progress)
        print("\nExtracting...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

        os.remove(zip_path)
        print(f"GloVe embeddings saved to {glove_file}")
        return glove_file

    except Exception as e:
        print(f"\nWarning: Could not download GloVe embeddings: {e}")
        print("Semantic evaluation will be skipped.")
        return None


def load_embeddings_matrix(vocab, embeddings_path: str, dim: int = 100):
    """Load GloVe vectors for words in vocab."""
    import numpy as np

    if not os.path.exists(embeddings_path):
        return None

    print(f"Loading embeddings from {embeddings_path}...")

    vocab_size = len(vocab)
    embeddings = np.random.uniform(-0.1, 0.1, (vocab_size, dim)).astype(np.float32)

    # Zero out padding
    embeddings[0] = np.zeros(dim)

    found = 0
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != dim + 1:
                continue
            word = parts[0]
            if word in vocab.word2idx:
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[vocab.word2idx[word]] = vector
                found += 1

    print(f"  Loaded {found}/{vocab_size} word embeddings")
    return embeddings


def save_results(results: dict, path: str):
    import json
    import numpy as np

    # Convert numpy types to native Python
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"Results saved to {path}")


def load_results(path: str) -> dict:
    import json
    with open(path, 'r') as f:
        return json.load(f)
