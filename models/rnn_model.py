# Vanilla RNN language model

import math
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .base_model import BaseLanguageModel


class SequenceDataset(Dataset):
    def __init__(self, sequences: List[List[int]], max_len: int = 50):
        self.data = [seq[:max_len] for seq in sequences]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        # input is seq[:-1], target is seq[1:] (predict next word)
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)


def collate_sequences(batch):
    """Pad sequences to same length in batch."""
    inputs, targets = zip(*batch)
    lengths = torch.tensor([len(x) for x in inputs])
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded, lengths


class RNNModel(nn.Module, BaseLanguageModel):

    def __init__(self, vocab_size: int, embedding_dim: int = 300,
                 hidden_size: int = 512, num_layers: int = 2,
                 dropout: float = 0.3, **kwargs):
        nn.Module.__init__(self)
        BaseLanguageModel.__init__(self, vocab_size)

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(
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
        output, hidden = self.rnn(embeds, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

    def train_model(self, train_data: List[List[int]], val_data: Optional[List[List[int]]] = None,
                    epochs: int = 20, batch_size: int = 64, learning_rate: float = 0.001,
                    grad_clip: float = 5.0, early_stopping_patience: int = 5, **kwargs) -> Dict:
        self.to(self.device)

        # Prepare data
        train_dataset = SequenceDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_sequences)

        val_loader = None
        if val_data:
            val_dataset = SequenceDataset(val_data)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)

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
            total_tokens = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for inputs, targets, lengths in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_size_actual = inputs.size(0)

                hidden = self.init_hidden(batch_size_actual)

                optimizer.zero_grad()
                outputs, _ = self(inputs, hidden)

                # Reshape for loss calculation
                outputs = outputs.view(-1, self.vocab_size)
                targets = targets.view(-1)

                loss = criterion(outputs, targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()

                mask = targets != 0
                total_loss += loss.item() * mask.sum().item()
                total_tokens += mask.sum().item()

                pbar.set_postfix({'loss': total_loss / total_tokens if total_tokens > 0 else 0})

            avg_train_loss = total_loss / total_tokens if total_tokens > 0 else 0
            history['train_loss'].append(avg_train_loss)

            # Validation
            if val_loader:
                val_loss = self._evaluate_loss(val_loader, criterion)
                val_perplexity = math.exp(val_loss) if val_loss < 100 else float('inf')
                history['val_loss'].append(val_loss)
                history['val_perplexity'].append(val_perplexity)

                scheduler.step(val_loss)

                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val PPL={val_perplexity:.2f}")

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

    def _evaluate_loss(self, data_loader: DataLoader, criterion: nn.Module) -> float:
        self.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for inputs, targets, lengths in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_size_actual = inputs.size(0)

                hidden = self.init_hidden(batch_size_actual)
                outputs, _ = self(inputs, hidden)

                outputs = outputs.view(-1, self.vocab_size)
                targets = targets.view(-1)

                mask = targets != 0
                if mask.sum() > 0:
                    loss = criterion(outputs[mask], targets[mask])
                    total_loss += loss.item() * mask.sum().item()
                    total_tokens += mask.sum().item()

        return total_loss / total_tokens if total_tokens > 0 else 0

    def predict_next(self, context: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        self.eval()

        with torch.no_grad():
            input_tensor = torch.tensor([context], dtype=torch.long).to(self.device)
            hidden = self.init_hidden(1)

            logits, _ = self(input_tensor, hidden)
            logits = logits[0, -1]  # Get last time step
            probs = torch.softmax(logits, dim=-1)

            top_probs, top_indices = torch.topk(probs, top_k)

        return [(idx.item(), prob.item()) for idx, prob in zip(top_indices, top_probs)]

    def get_probabilities(self, context: List[int]) -> torch.Tensor:
        self.eval()

        with torch.no_grad():
            input_tensor = torch.tensor([context], dtype=torch.long).to(self.device)
            hidden = self.init_hidden(1)

            logits, _ = self(input_tensor, hidden)
            logits = logits[0, -1]
            probs = torch.softmax(logits, dim=-1)

        return probs.cpu()

    def calculate_perplexity(self, sequences: List[List[int]]) -> float:
        dataset = SequenceDataset(sequences)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_sequences)
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

        self.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for inputs, targets, lengths in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_size_actual = inputs.size(0)

                hidden = self.init_hidden(batch_size_actual)
                outputs, _ = self(inputs, hidden)

                outputs = outputs.view(-1, self.vocab_size)
                targets = targets.view(-1)

                mask = targets != 0
                if mask.sum() > 0:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    total_tokens += mask.sum().item()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        return math.exp(avg_loss) if avg_loss < 100 else float('inf')

    def save(self, path: str):
        data = {
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
    def load(cls, path: str) -> 'RNNModel':
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
        info.update({
            'model_architecture': 'RNN',
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout_rate,
            'num_parameters': sum(p.numel() for p in self.parameters()),
        })
        return info
