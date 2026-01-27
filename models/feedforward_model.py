# Feedforward neural network for language modeling

import math
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base_model import BaseLanguageModel


class FeedforwardLM(nn.Module, BaseLanguageModel):
    """Simple feedforward network that uses a fixed context window."""

    def __init__(self, vocab_size: int, embedding_dim: int = 300,
                 hidden_size: int = 512, context_size: int = 10,
                 dropout: float = 0.3, num_layers: int = 2, **kwargs):
        nn.Module.__init__(self)
        BaseLanguageModel.__init__(self, vocab_size)

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.dropout_rate = dropout
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Hidden layers
        layers = []
        input_size = embedding_dim * context_size

        for i in range(num_layers):
            layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        # For tracking training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(x)
        embeds = embeds.view(embeds.size(0), -1)  # flatten context embeddings
        hidden = self.hidden_layers(embeds)
        logits = self.fc_out(hidden)
        return logits

    def train_model(self, train_data: List[List[int]], val_data: Optional[List[List[int]]] = None,
                    epochs: int = 20, batch_size: int = 64, learning_rate: float = 0.001,
                    grad_clip: float = 5.0, early_stopping_patience: int = 5, **kwargs) -> Dict:
        self.to(self.device)
        self.train()

        # Prepare data
        train_loader = self._prepare_data(train_data, batch_size, shuffle=True)
        val_loader = self._prepare_data(val_data, batch_size, shuffle=False) if val_data else None

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        history = {'train_loss': [], 'val_loss': [], 'val_perplexity': []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.train()
            total_loss = 0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': total_loss / num_batches})

            avg_train_loss = total_loss / num_batches
            history['train_loss'].append(avg_train_loss)

            # Validation
            if val_loader:
                val_loss = self._evaluate_loss(val_loader, criterion)
                val_perplexity = math.exp(val_loss)
                history['val_loss'].append(val_loss)
                history['val_perplexity'].append(val_perplexity)

                scheduler.step(val_loss)

                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val PPL={val_perplexity:.2f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}")

        self.is_trained = True
        return history

    def _prepare_data(self, sequences: List[List[int]], batch_size: int,
                      shuffle: bool = True) -> DataLoader:
        inputs = []
        targets = []

        for seq in sequences:
            # Pad sequence at the beginning
            padded = [0] * self.context_size + seq

            for i in range(len(seq)):
                context = padded[i:i + self.context_size]
                target = seq[i]
                inputs.append(context)
                targets.append(target)

        inputs_tensor = torch.tensor(inputs, dtype=torch.long)
        targets_tensor = torch.tensor(targets, dtype=torch.long)

        dataset = TensorDataset(inputs_tensor, targets_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _evaluate_loss(self, data_loader: DataLoader, criterion: nn.Module) -> float:
        self.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def predict_next(self, context: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        self.eval()

        # Pad or truncate context
        if len(context) < self.context_size:
            context = [0] * (self.context_size - len(context)) + context
        else:
            context = context[-self.context_size:]

        with torch.no_grad():
            input_tensor = torch.tensor([context], dtype=torch.long).to(self.device)
            logits = self(input_tensor)
            probs = torch.softmax(logits, dim=-1)[0]

            top_probs, top_indices = torch.topk(probs, top_k)

        return [(idx.item(), prob.item()) for idx, prob in zip(top_indices, top_probs)]

    def get_probabilities(self, context: List[int]) -> torch.Tensor:
        self.eval()

        if len(context) < self.context_size:
            context = [0] * (self.context_size - len(context)) + context
        else:
            context = context[-self.context_size:]

        with torch.no_grad():
            input_tensor = torch.tensor([context], dtype=torch.long).to(self.device)
            logits = self(input_tensor)
            probs = torch.softmax(logits, dim=-1)[0]

        return probs.cpu()

    def calculate_perplexity(self, sequences: List[List[int]]) -> float:
        self.eval()
        data_loader = self._prepare_data(sequences, batch_size=64, shuffle=False)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self(inputs)

                # Calculate loss for non-padding tokens
                mask = targets != 0
                if mask.sum() > 0:
                    loss = criterion(outputs[mask], targets[mask])
                    total_loss += loss.item() * mask.sum().item()
                    total_tokens += mask.sum().item()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        return math.exp(avg_loss)

    def save(self, path: str):
        data = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'context_size': self.context_size,
            'dropout_rate': self.dropout_rate,
            'num_layers': self.num_layers,
            'state_dict': self.state_dict(),
            'is_trained': self.is_trained,
        }
        torch.save(data, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'FeedforwardLM':
        data = torch.load(path, map_location='cpu')

        model = cls(
            vocab_size=data['vocab_size'],
            embedding_dim=data['embedding_dim'],
            hidden_size=data['hidden_size'],
            context_size=data['context_size'],
            dropout=data['dropout_rate'],
            num_layers=data['num_layers']
        )
        model.load_state_dict(data['state_dict'])
        model.is_trained = data.get('is_trained', True)

        print(f"Model loaded from {path}")
        return model

    def get_model_info(self) -> Dict:
        info = super().get_model_info()
        info.update({
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'context_size': self.context_size,
            'dropout': self.dropout_rate,
            'num_layers': self.num_layers,
            'num_parameters': sum(p.numel() for p in self.parameters()),
        })
        return info
