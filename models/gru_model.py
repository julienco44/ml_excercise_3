# GRU language model - simpler alternative to LSTM

import math
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .rnn_model import RNNModel, SequenceDataset, collate_sequences


class GRUModel(RNNModel):

    def __init__(self, vocab_size: int, embedding_dim: int = 300,
                 hidden_size: int = 512, num_layers: int = 2,
                 dropout: float = 0.3, **kwargs):
        # skip RNN's __init__
        nn.Module.__init__(self)
        from .base_model import BaseLanguageModel
        BaseLanguageModel.__init__(self, vocab_size)

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim, hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        embeds = self.embedding(x)
        output, hidden = self.gru(embeds, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

    def save(self, path: str):
        data = {
            'model_type': 'GRU',
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'state_dict': self.state_dict(),
            'is_trained': self.is_trained,
        }
        torch.save(data, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'GRUModel':
        data = torch.load(path, map_location='cpu')

        model = cls(
            vocab_size=data['vocab_size'],
            embedding_dim=data['embedding_dim'],
            hidden_size=data['hidden_size'],
            num_layers=data['num_layers'],
            dropout=data['dropout_rate']
        )
        model.load_state_dict(data['state_dict'])
        model.is_trained = data.get('is_trained', True)

        print(f"Model loaded from {path}")
        return model

    def get_model_info(self) -> Dict:
        info = super().get_model_info()
        info['model_architecture'] = 'GRU'
        return info
