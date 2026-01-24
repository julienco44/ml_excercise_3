import pickle
from collections import Counter
from typing import List, Dict


class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<SOS>": 2,
            "<EOS>": 3
        }
        self.idx2word: Dict[int, str] = {
            0: "<PAD>",
            1: "<UNK>",
            2: "<SOS>",
            3: "<EOS>"
        }
        self.word_freq = Counter()

    def build(self, sentences: List[List[str]], max_size: int = None):
        print("Counting word frequencies...")
        for sentence in sentences:
            self.word_freq.update(sentence)

        print(f"Total unique words: {len(self.word_freq)}")

        words = [w for w, c in self.word_freq.most_common() if c >= self.min_freq]
        print(f"Words with freq >= {self.min_freq}: {len(words)}")

        if max_size:
            words = words[:max_size - len(self.word2idx)]

        for word in words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"Final vocabulary size: {len(self.word2idx)}")

    def encode(self, sentence: List[str]) -> List[int]:
        return [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in sentence]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2word.get(i, "<UNK>") for i in indices]

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: str):
        data = {
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "word_freq": dict(self.word_freq),
            "min_freq": self.min_freq
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Vocabulary saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = data["idx2word"]
        self.word_freq = Counter(data.get("word_freq", {}))
        self.min_freq = data.get("min_freq", 2)
        print(f"Vocabulary loaded from {path} ({len(self)} words)")

    def get_stats(self) -> Dict:
        return {
            "vocab_size": len(self.word2idx),
            "total_words_seen": sum(self.word_freq.values()),
            "unique_words_seen": len(self.word_freq),
            "min_freq": self.min_freq,
            "most_common": self.word_freq.most_common(20)
        }

    def load_embeddings(self, path: str, dim: int = 300) -> "torch.Tensor":
        """
        Loads pretrained embeddings (GloVe/FastText format) for the current vocabulary.
        Returns a torch tensor of shape (vocab_size, dim).
        """
        print(f"Loading embeddings from {path}...")
        try:
            import numpy as np
            import torch
        except ImportError:
            print("numpy/torch not available for embeddings loading")
            return None

        embeddings = np.random.uniform(-0.1, 0.1, (len(self), dim))
        
        # Zero out <PAD>
        if "<PAD>" in self.word2idx:
            embeddings[self.word2idx["<PAD>"]] = np.zeros(dim)

        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < dim + 1:
                    continue
                word = parts[0]
                if word in self.word2idx:
                    vector = np.array(parts[1:], dtype=np.float32)
                    if vector.shape[0] == dim:
                        embeddings[self.word2idx[word]] = vector
                        count += 1
        
        print(f"Loaded {count}/{len(self)} embeddings from pretrained file")
        return torch.tensor(embeddings, dtype=torch.float32)
