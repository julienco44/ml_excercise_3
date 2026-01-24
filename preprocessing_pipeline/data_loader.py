import os
import csv
from typing import List
from sklearn.datasets import fetch_20newsgroups


class DataLoader:
    @staticmethod
    def load_20newsgroups() -> List[str]:
        print("Loading 20 Newsgroups dataset (auto-download)...")
        newsgroups = fetch_20newsgroups(
            subset='all',
            remove=('headers', 'footers', 'quotes')
        )
        print(f"Loaded {len(newsgroups.data)} documents")
        return newsgroups.data

    @staticmethod
    def load_fake_news(filepath: str) -> List[str]:
        print(f"Loading Fake News dataset from {filepath}...")
        texts = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text", "")
                if text:
                    texts.append(text)
        print(f"Loaded {len(texts)} documents")
        return texts

    @staticmethod
    def load_leipzig_file(filepath: str) -> List[str]:
        print(f"Loading Leipzig corpus from {filepath}...")
        texts = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                text = parts[1] if len(parts) >= 2 else parts[0]
                texts.append(text)

                if line_num % 100000 == 0:
                    print(f"  Read {line_num} lines...")

        print(f"Loaded {len(texts)} sentences")
        return texts

    @staticmethod
    def load_plain_text(filepath: str) -> List[str]:
        print(f"Loading plain text from {filepath}...")
        texts = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        print(f"Loaded {len(texts)} lines")
        return texts

    @staticmethod
    def load_from_directory(directory: str, extension: str = ".txt") -> List[str]:
        print(f"Loading all {extension} files from {directory}...")
        texts = []
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith(extension):
                    filepath = os.path.join(root, filename)
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        texts.append(content)
        print(f"Loaded {len(texts)} files")
        return texts


def get_dataset(name: str, path: str = None) -> List[str]:
    loader = DataLoader()

    if name == "20newsgroups":
        return loader.load_20newsgroups()
    elif name == "fake_news":
        if not path:
            raise ValueError("Path required for fake_news dataset")
        return loader.load_fake_news(path)
    elif name == "leipzig":
        if not path:
            raise ValueError("Path required for leipzig dataset")
        return loader.load_leipzig_file(path)
    elif name == "plain":
        if not path:
            raise ValueError("Path required for plain text")
        return loader.load_plain_text(path)
    elif name == "directory":
        if not path:
            raise ValueError("Path required for directory loading")
        return loader.load_from_directory(path)
    else:
        raise ValueError(f"Unknown dataset: {name}")
