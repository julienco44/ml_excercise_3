# N-gram model with different smoothing methods

import pickle
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import numpy as np
import torch

from .base_model import BaseLanguageModel


class NgramModel(BaseLanguageModel):
    """N-gram language model with Laplace, Kneser-Ney, or interpolation smoothing."""

    def __init__(self, vocab_size: int, n: int = 3, smoothing: str = "laplace",
                 alpha: float = 1.0, **kwargs):
        super().__init__(vocab_size)
        self.n = n
        self.smoothing = smoothing
        self.alpha = alpha

        # N-gram counts: context (tuple) -> next_word -> count
        self.ngram_counts: Dict[Tuple, Counter] = defaultdict(Counter)
        # Context counts: context (tuple) -> total count
        self.context_counts: Counter = Counter()
        # Unigram counts for backoff/interpolation
        self.unigram_counts: Counter = Counter()
        # Vocabulary set
        self.vocab: set = set()
        # Total tokens seen
        self.total_tokens: int = 0

        # For Kneser-Ney smoothing
        self.discount: float = 0.75
        self.continuation_counts: Counter = Counter()
        self.unique_continuations: Dict[Tuple, int] = defaultdict(int)

        # For interpolation smoothing
        self.lower_order_models: Dict[int, 'NgramModel'] = {}

    def train_model(self, train_data: List[List[int]], val_data=None, **kwargs) -> Dict:
        """Train on token sequences and count n-grams."""
        print(f"Training {self.n}-gram model with {self.smoothing} smoothing...")

        # Build vocabulary from sequences
        for seq in train_data:
            self.vocab.update(seq)
            self.unigram_counts.update(seq)
            self.total_tokens += len(seq)

        # Count n-grams
        for seq in train_data:
            padded_seq = [0] * (self.n - 1) + seq  # 0 is <PAD>

            for i in range(len(padded_seq) - self.n + 1):
                context = tuple(padded_seq[i:i + self.n - 1])
                next_word = padded_seq[i + self.n - 1]

                self.ngram_counts[context][next_word] += 1
                self.context_counts[context] += 1

                if self.smoothing == "kneser_ney":
                    self.continuation_counts[next_word] += 1
                    self.unique_continuations[context] += 1

        # For interpolation, train lower-order models
        if self.smoothing == "interpolation" and self.n > 1:
            for order in range(1, self.n):
                self.lower_order_models[order] = NgramModel(
                    vocab_size=self.vocab_size, n=order,
                    smoothing="laplace", alpha=self.alpha
                )
                self.lower_order_models[order].train_model(train_data)

        self.is_trained = True

        history = {
            'num_sequences': len(train_data),
            'unique_ngrams': len(self.ngram_counts),
            'vocab_size': self.vocab_size,
        }

        if val_data:
            history['val_perplexity'] = self.calculate_perplexity(val_data)

        print(f"  Trained on {len(train_data)} sequences")
        print(f"  Unique {self.n}-grams: {len(self.ngram_counts)}")

        return history

    def _get_laplace_prob(self, context: Tuple, word: int) -> float:
        # add-alpha smoothing
        count = self.ngram_counts[context][word]
        context_count = self.context_counts[context]
        return (count + self.alpha) / (context_count + self.alpha * self.vocab_size)

    def _get_kneser_ney_prob(self, context: Tuple, word: int) -> float:
        count = self.ngram_counts[context][word]
        context_count = self.context_counts[context]

        if context_count == 0:
            total_cont = sum(self.continuation_counts.values())
            return self.continuation_counts[word] / total_cont if total_cont else 1.0 / self.vocab_size

        first_term = max(count - self.discount, 0) / context_count
        lambda_weight = (self.discount * self.unique_continuations[context]) / context_count

        total_cont = sum(self.continuation_counts.values())
        lower_prob = self.continuation_counts[word] / total_cont if total_cont else 1.0 / self.vocab_size

        return first_term + lambda_weight * lower_prob

    def _get_interpolation_prob(self, context: Tuple, word: int) -> float:
        # mix probabilities from different n-gram orders
        lambdas = [1.0 / self.n] * self.n
        prob = 0.0

        for order in range(1, self.n + 1):
            if order == self.n:
                count = self.ngram_counts[context][word]
                context_count = self.context_counts[context]
                order_prob = count / context_count if context_count > 0 else 0
            elif order == 1:
                order_prob = self.unigram_counts[word] / self.total_tokens if self.total_tokens > 0 else 1.0 / self.vocab_size
            else:
                shorter_context = context[-(order-1):]
                model = self.lower_order_models.get(order)
                if model:
                    order_prob = model._get_probability(shorter_context, word)
                else:
                    order_prob = 0

            prob += lambdas[order - 1] * order_prob

        return prob if prob > 0 else 1e-10

    def _get_probability(self, context: Tuple, word: int) -> float:
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
        if len(context) >= self.n - 1:
            context_tuple = tuple(context[-(self.n - 1):])
        else:
            context_tuple = tuple([0] * (self.n - 1 - len(context)) + context)

        # Fast prediction using only seen n-grams
        seen_words = self.ngram_counts.get(context_tuple, Counter())

        if not seen_words:
            top_unigrams = self.unigram_counts.most_common(top_k)
            total = sum(self.unigram_counts.values())
            return [(w, c / total) for w, c in top_unigrams]

        word_probs = []
        for word, count in seen_words.items():
            prob = self._get_probability(context_tuple, word)
            word_probs.append((word, prob))

        word_probs.sort(key=lambda x: x[1], reverse=True)
        return word_probs[:top_k]

    def get_probabilities(self, context: List[int]) -> torch.Tensor:
        probs = torch.zeros(self.vocab_size)

        if len(context) >= self.n - 1:
            context_tuple = tuple(context[-(self.n - 1):])
        else:
            context_tuple = tuple([0] * (self.n - 1 - len(context)) + context)

        for word in range(self.vocab_size):
            probs[word] = self._get_probability(context_tuple, word)

        # Normalize
        probs = probs / probs.sum()
        return probs

    def calculate_perplexity(self, sequences: List[List[int]]) -> float:
        total_log_prob = 0.0
        total_words = 0

        for seq in sequences:
            padded_seq = [0] * (self.n - 1) + seq

            for i in range(len(padded_seq) - self.n + 1):
                context = tuple(padded_seq[i:i + self.n - 1])
                word = padded_seq[i + self.n - 1]

                prob = self._get_probability(context, word)
                if prob > 0:
                    total_log_prob += math.log2(prob)
                else:
                    total_log_prob += math.log2(1e-10)

                total_words += 1

        avg_log_prob = total_log_prob / total_words if total_words > 0 else 0
        return 2 ** (-avg_log_prob)

    def save(self, path: str):
        data = {
            'vocab_size': self.vocab_size,
            'n': self.n,
            'smoothing': self.smoothing,
            'alpha': self.alpha,
            'ngram_counts': dict(self.ngram_counts),
            'context_counts': dict(self.context_counts),
            'unigram_counts': dict(self.unigram_counts),
            'vocab': self.vocab,
            'total_tokens': self.total_tokens,
            'discount': self.discount,
            'continuation_counts': dict(self.continuation_counts),
            'unique_continuations': dict(self.unique_continuations),
            'is_trained': self.is_trained,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'NgramModel':
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls(
            vocab_size=data['vocab_size'],
            n=data['n'],
            smoothing=data['smoothing'],
            alpha=data['alpha']
        )
        model.ngram_counts = defaultdict(Counter, {k: Counter(v) for k, v in data['ngram_counts'].items()})
        model.context_counts = Counter(data['context_counts'])
        model.unigram_counts = Counter(data['unigram_counts'])
        model.vocab = data['vocab']
        model.total_tokens = data['total_tokens']
        model.discount = data.get('discount', 0.75)
        model.continuation_counts = Counter(data.get('continuation_counts', {}))
        model.unique_continuations = defaultdict(int, data.get('unique_continuations', {}))
        model.is_trained = data.get('is_trained', True)

        print(f"Model loaded from {path}")
        return model

    def get_model_info(self) -> Dict:
        info = super().get_model_info()
        info.update({
            'n': self.n,
            'smoothing': self.smoothing,
            'alpha': self.alpha,
            'unique_ngrams': len(self.ngram_counts),
            'total_tokens': self.total_tokens,
        })
        return info
