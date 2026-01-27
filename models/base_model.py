# Base class for language models

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import torch


class BaseLanguageModel(ABC):
    """Base class that all models inherit from."""

    def __init__(self, vocab_size: int, **kwargs):
        self.vocab_size = vocab_size
        self.is_trained = False

    @abstractmethod
    def train_model(self, train_data, val_data=None, **kwargs):
        """Train the model. Returns training history dict."""
        pass

    @abstractmethod
    def predict_next(self, context: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        """Return top-k predictions as (word_idx, prob) tuples."""
        pass

    @abstractmethod
    def get_probabilities(self, context: List[int]) -> torch.Tensor:
        """Get probability distribution over vocabulary."""
        pass

    @abstractmethod
    def calculate_perplexity(self, sequences: List[List[int]]) -> float:
        """Calculate perplexity on sequences."""
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseLanguageModel':
        pass

    def get_model_info(self) -> Dict:
        return {
            'model_type': self.__class__.__name__,
            'vocab_size': self.vocab_size,
            'is_trained': self.is_trained,
        }
