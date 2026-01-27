"""
N-gram Language Model for Next Word Prediction

Supports:
- Various n-gram sizes (bigram, trigram, etc.)
- Multiple smoothing techniques (Laplace, Kneser-Ney, Interpolation)
- Top-k predictions
"""

import pickle
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import numpy as np


class NGramModel:
    """N-gram language model with multiple smoothing options."""

    def __init__(self, n: int = 3, smoothing: str = "laplace", alpha: float = 1.0):
        """
        Initialize N-gram model.

        Args:
            n: Order of the n-gram (2=bigram, 3=trigram, etc.)
            smoothing: Smoothing technique ('none', 'laplace', 'kneser_ney', 'interpolation')
            alpha: Smoothing parameter (for Laplace smoothing)
        """
        self.n = n
        self.smoothing = smoothing
        self.alpha = alpha

        # N-gram counts: context (tuple) -> next_word -> count
        self.ngram_counts: Dict[Tuple, Counter] = defaultdict(Counter)
        # Context counts: context (tuple) -> total count
        self.context_counts: Counter = Counter()
        # Unigram counts for backoff/interpolation
        self.unigram_counts: Counter = Counter()
        # Vocabulary
        self.vocab: set = set()
        self.vocab_size: int = 0
        # Total tokens seen
        self.total_tokens: int = 0

        # For Kneser-Ney smoothing
        self.discount: float = 0.75
        self.continuation_counts: Counter = Counter()  # How many different contexts precede word
        self.unique_continuations: Dict[Tuple, int] = defaultdict(int)  # Unique words following context

        # For interpolation smoothing - lower order models
        self.lower_order_models: Dict[int, 'NGramModel'] = {}

    def train(self, sequences: List[List[int]], vocab_size: int = None):
        """
        Train the n-gram model on sequences of token indices.

        Args:
            sequences: List of sequences (each sequence is a list of token indices)
            vocab_size: Size of vocabulary (for smoothing)
        """
        print(f"Training {self.n}-gram model with {self.smoothing} smoothing...")

        # Build vocabulary from sequences
        for seq in sequences:
            self.vocab.update(seq)
            self.unigram_counts.update(seq)
            self.total_tokens += len(seq)

        self.vocab_size = vocab_size if vocab_size else len(self.vocab)

        # Count n-grams
        for seq in sequences:
            # Add padding for context at the beginning
            padded_seq = [0] * (self.n - 1) + seq  # 0 is <PAD>

            for i in range(len(padded_seq) - self.n + 1):
                context = tuple(padded_seq[i:i + self.n - 1])
                next_word = padded_seq[i + self.n - 1]

                self.ngram_counts[context][next_word] += 1
                self.context_counts[context] += 1

                # For Kneser-Ney: track continuation counts
                if self.smoothing == "kneser_ney":
                    self.continuation_counts[next_word] += 1
                    self.unique_continuations[context] += 1

        # For interpolation, train lower-order models
        if self.smoothing == "interpolation" and self.n > 1:
            for order in range(1, self.n):
                self.lower_order_models[order] = NGramModel(n=order, smoothing="laplace", alpha=self.alpha)
                self.lower_order_models[order].train(sequences, vocab_size)

        print(f"  Trained on {len(sequences)} sequences")
        print(f"  Unique {self.n}-grams: {len(self.ngram_counts)}")
        print(f"  Vocabulary size: {self.vocab_size}")

    def _get_laplace_prob(self, context: Tuple, word: int) -> float:
        """Laplace (add-alpha) smoothed probability."""
        count = self.ngram_counts[context][word]
        context_count = self.context_counts[context]
        return (count + self.alpha) / (context_count + self.alpha * self.vocab_size)

    def _get_kneser_ney_prob(self, context: Tuple, word: int) -> float:
        """Kneser-Ney smoothed probability."""
        count = self.ngram_counts[context][word]
        context_count = self.context_counts[context]

        if context_count == 0:
            # Backoff to unigram continuation probability
            return self.continuation_counts[word] / sum(self.continuation_counts.values()) if self.continuation_counts else 1.0 / self.vocab_size

        # Discounted probability
        first_term = max(count - self.discount, 0) / context_count

        # Interpolation weight
        lambda_weight = (self.discount * self.unique_continuations[context]) / context_count

        # Lower order probability (unigram continuation)
        lower_prob = self.continuation_counts[word] / sum(self.continuation_counts.values()) if self.continuation_counts else 1.0 / self.vocab_size

        return first_term + lambda_weight * lower_prob

    def _get_interpolation_prob(self, context: Tuple, word: int) -> float:
        """Interpolated probability mixing different n-gram orders."""
        # Weights for each order (can be tuned)
        lambdas = self._get_interpolation_weights()

        prob = 0.0
        for order in range(1, self.n + 1):
            if order == self.n:
                # Current model
                count = self.ngram_counts[context][word]
                context_count = self.context_counts[context]
                if context_count > 0:
                    order_prob = count / context_count
                else:
                    order_prob = 0
            elif order == 1:
                # Unigram
                order_prob = self.unigram_counts[word] / self.total_tokens if self.total_tokens > 0 else 1.0 / self.vocab_size
            else:
                # Lower order model
                shorter_context = context[-(order-1):]
                model = self.lower_order_models.get(order)
                if model:
                    order_prob = model.get_probability(shorter_context, word)
                else:
                    order_prob = 0

            prob += lambdas[order - 1] * order_prob

        return prob if prob > 0 else 1e-10

    def _get_interpolation_weights(self) -> List[float]:
        """Get interpolation weights for each n-gram order."""
        # Simple uniform weights - can be optimized with held-out data
        weights = [1.0 / self.n] * self.n
        return weights

    def get_probability(self, context: Tuple, word: int) -> float:
        """
        Get probability of word given context.

        Args:
            context: Tuple of preceding token indices
            word: Target word index

        Returns:
            Probability P(word | context)
        """
        # Ensure context is the right length
        if len(context) < self.n - 1:
            context = (0,) * (self.n - 1 - len(context)) + context
        elif len(context) > self.n - 1:
            context = context[-(self.n - 1):]

        if self.smoothing == "none":
            count = self.ngram_counts[context][word]
            context_count = self.context_counts[context]
            return count / context_count if context_count > 0 else 0
        elif self.smoothing == "laplace":
            return self._get_laplace_prob(context, word)
        elif self.smoothing == "kneser_ney":
            return self._get_kneser_ney_prob(context, word)
        elif self.smoothing == "interpolation":
            return self._get_interpolation_prob(context, word)
        else:
            raise ValueError(f"Unknown smoothing method: {self.smoothing}")

    def predict_next(self, context: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Predict the most likely next words given context.

        Args:
            context: List of preceding token indices
            top_k: Number of top predictions to return

        Returns:
            List of (word_index, probability) tuples sorted by probability
        """
        context_tuple = tuple(context[-(self.n - 1):]) if len(context) >= self.n - 1 else tuple([0] * (self.n - 1 - len(context)) + context)

        # Get probabilities for all words in vocabulary
        word_probs = []
        for word in self.vocab:
            prob = self.get_probability(context_tuple, word)
            word_probs.append((word, prob))

        # Sort by probability and return top-k
        word_probs.sort(key=lambda x: x[1], reverse=True)
        return word_probs[:top_k]

    def predict_next_fast(self, context: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Fast prediction using only seen n-grams (no smoothing over full vocab).
        More efficient for large vocabularies.
        """
        context_tuple = tuple(context[-(self.n - 1):]) if len(context) >= self.n - 1 else tuple([0] * (self.n - 1 - len(context)) + context)

        # Get words that have been seen after this context
        seen_words = self.ngram_counts.get(context_tuple, Counter())

        if not seen_words:
            # Backoff to unigram
            top_unigrams = self.unigram_counts.most_common(top_k)
            total = sum(self.unigram_counts.values())
            return [(w, c / total) for w, c in top_unigrams]

        word_probs = []
        for word, count in seen_words.items():
            prob = self.get_probability(context_tuple, word)
            word_probs.append((word, prob))

        word_probs.sort(key=lambda x: x[1], reverse=True)
        return word_probs[:top_k]

    def calculate_perplexity(self, sequences: List[List[int]]) -> float:
        """
        Calculate perplexity on a set of sequences.

        Args:
            sequences: List of token sequences

        Returns:
            Perplexity score (lower is better)
        """
        total_log_prob = 0.0
        total_words = 0

        for seq in sequences:
            padded_seq = [0] * (self.n - 1) + seq

            for i in range(len(padded_seq) - self.n + 1):
                context = tuple(padded_seq[i:i + self.n - 1])
                word = padded_seq[i + self.n - 1]

                prob = self.get_probability(context, word)
                if prob > 0:
                    total_log_prob += math.log2(prob)
                else:
                    total_log_prob += math.log2(1e-10)  # Small value to avoid -inf

                total_words += 1

        avg_log_prob = total_log_prob / total_words if total_words > 0 else 0
        perplexity = 2 ** (-avg_log_prob)

        return perplexity

    def save(self, path: str):
        """Save model to file."""
        data = {
            'n': self.n,
            'smoothing': self.smoothing,
            'alpha': self.alpha,
            'ngram_counts': dict(self.ngram_counts),
            'context_counts': dict(self.context_counts),
            'unigram_counts': dict(self.unigram_counts),
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'total_tokens': self.total_tokens,
            'discount': self.discount,
            'continuation_counts': dict(self.continuation_counts),
            'unique_continuations': dict(self.unique_continuations),
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'NGramModel':
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls(n=data['n'], smoothing=data['smoothing'], alpha=data['alpha'])
        model.ngram_counts = defaultdict(Counter, {k: Counter(v) for k, v in data['ngram_counts'].items()})
        model.context_counts = Counter(data['context_counts'])
        model.unigram_counts = Counter(data['unigram_counts'])
        model.vocab = data['vocab']
        model.vocab_size = data['vocab_size']
        model.total_tokens = data['total_tokens']
        model.discount = data.get('discount', 0.75)
        model.continuation_counts = Counter(data.get('continuation_counts', {}))
        model.unique_continuations = defaultdict(int, data.get('unique_continuations', {}))

        print(f"Model loaded from {path}")
        return model

    def get_stats(self) -> Dict:
        """Get model statistics."""
        return {
            'n': self.n,
            'smoothing': self.smoothing,
            'vocab_size': self.vocab_size,
            'total_tokens': self.total_tokens,
            'unique_ngrams': len(self.ngram_counts),
            'unique_contexts': len(self.context_counts),
        }
