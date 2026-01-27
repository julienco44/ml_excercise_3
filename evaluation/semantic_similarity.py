# Semantic similarity using word embeddings (GloVe)

import os
from typing import List, Dict, Tuple, Optional
import numpy as np


class SemanticEvaluator:
    """Use word embeddings to check if predictions are semantically similar."""

    def __init__(self, vocab, embeddings_path: Optional[str] = None,
                 embedding_dim: int = 100):
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.embeddings = None
        self.normalized_embeddings = None

        if embeddings_path and os.path.exists(embeddings_path):
            self.load_embeddings(embeddings_path, embedding_dim)

    def load_embeddings(self, path: str, dim: int = 100):
        print(f"Loading embeddings from {path}...")

        vocab_size = len(self.vocab.word2idx) if hasattr(self.vocab, 'word2idx') else len(self.vocab)
        self.embeddings = np.random.uniform(-0.1, 0.1, (vocab_size, dim)).astype(np.float32)

        # Zero out padding
        self.embeddings[0] = np.zeros(dim)

        found = 0
        word2idx = self.vocab.word2idx if hasattr(self.vocab, 'word2idx') else self.vocab

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != dim + 1:
                    continue
                word = parts[0]
                if word in word2idx:
                    vector = np.array(parts[1:], dtype=np.float32)
                    self.embeddings[word2idx[word]] = vector
                    found += 1

        print(f"  Loaded {found}/{vocab_size} word embeddings")

        # Precompute normalized embeddings
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.normalized_embeddings = self.embeddings / norms

    def cosine_similarity(self, word1_idx: int, word2_idx: int) -> float:
        if self.normalized_embeddings is None:
            return 0.0

        if word1_idx >= len(self.normalized_embeddings) or word2_idx >= len(self.normalized_embeddings):
            return 0.0

        return float(np.dot(self.normalized_embeddings[word1_idx],
                           self.normalized_embeddings[word2_idx]))

    def semantic_accuracy(self, predictions: List[int], targets: List[int],
                         threshold: float = 0.7) -> float:
        """Count as correct if exact match or cosine sim >= threshold."""
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded")

        correct = 0
        valid = 0

        for pred, target in zip(predictions, targets):
            if pred < 4 or target < 4:
                continue

            if pred == target:
                correct += 1
            else:
                similarity = self.cosine_similarity(pred, target)
                if similarity >= threshold:
                    correct += 1
            valid += 1

        return correct / valid if valid > 0 else 0.0

    def average_semantic_similarity(self, predictions: List[int],
                                   targets: List[int]) -> float:
        if self.embeddings is None:
            return 0.0

        similarities = []
        for pred, target in zip(predictions, targets):
            if pred < 4 or target < 4:
                continue
            sim = self.cosine_similarity(pred, target)
            similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

    def top_k_semantic_accuracy(self, predictions_list: List[List[Tuple[int, float]]],
                               targets: List[int], k: int = 5,
                               threshold: float = 0.7) -> float:
        if self.embeddings is None:
            return 0.0

        correct = 0
        valid = 0

        for preds, target in zip(predictions_list, targets):
            if target < 4:
                continue

            top_k_words = [p[0] for p in preds[:k]]

            # Check exact match
            if target in top_k_words:
                correct += 1
                valid += 1
                continue

            # Check semantic similarity
            found = False
            for word_idx in top_k_words:
                if word_idx >= 4:
                    similarity = self.cosine_similarity(word_idx, target)
                    if similarity >= threshold:
                        found = True
                        break

            if found:
                correct += 1
            valid += 1

        return correct / valid if valid > 0 else 0.0

    def evaluate_semantic(self, predictions: List[int],
                         top_k_predictions: List[List[Tuple[int, float]]],
                         targets: List[int]) -> Dict:
        if self.embeddings is None:
            return {}

        return {
            'avg_semantic_similarity': self.average_semantic_similarity(predictions, targets),
            'semantic_accuracy_0.5': self.semantic_accuracy(predictions, targets, threshold=0.5),
            'semantic_accuracy_0.7': self.semantic_accuracy(predictions, targets, threshold=0.7),
            'top_5_semantic_accuracy': self.top_k_semantic_accuracy(top_k_predictions, targets, k=5, threshold=0.5),
        }

    def find_similar_words(self, word_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.normalized_embeddings is None:
            return []

        word_vec = self.normalized_embeddings[word_idx]
        similarities = np.dot(self.normalized_embeddings, word_vec)

        # Get top-k (excluding the word itself)
        top_indices = np.argsort(similarities)[::-1][:top_k + 1]
        results = [(idx, similarities[idx]) for idx in top_indices if idx != word_idx]

        return results[:top_k]
