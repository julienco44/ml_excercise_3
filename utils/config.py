# Configuration management for experiments

import os
import yaml
import json
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Config:

    # Experiment
    experiment_name: str = "experiment"
    model_type: str = "lstm"  # ngram, fnn, rnn, lstm, gru

    # Model hyperparameters
    vocab_size: int = 20000
    embedding_dim: int = 300
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.3
    context_size: int = 10  # For FNN

    # N-gram specific
    n: int = 3
    smoothing: str = "laplace"
    alpha: float = 1.0

    # Training hyperparameters
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 20
    grad_clip: float = 5.0
    early_stopping_patience: int = 5

    # Data paths
    data_dir: str = "processed_data"
    train_data_path: str = ""
    val_data_path: str = ""
    test_data_path: str = ""

    # Output paths
    output_dir: str = "experiments/results"
    checkpoint_dir: str = "experiments/checkpoints"
    checkpoint_path: str = ""

    # Embeddings
    embeddings_path: str = "embeddings/glove.6B.100d.txt"

    # Evaluation
    max_eval_samples: int = 10000
    top_k: int = 10

    def __post_init__(self):
        # set default paths
        if not self.train_data_path:
            self.train_data_path = os.path.join(self.data_dir, "train_dataset.pkl")
        if not self.val_data_path:
            self.val_data_path = os.path.join(self.data_dir, "val_dataset.pkl")
        if not self.test_data_path:
            self.test_data_path = os.path.join(self.data_dir, "test_dataset.pkl")
        if not self.checkpoint_path:
            self.checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}.pt")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k) or k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_name': self.experiment_name,
            'model_type': self.model_type,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'context_size': self.context_size,
            'n': self.n,
            'smoothing': self.smoothing,
            'alpha': self.alpha,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'grad_clip': self.grad_clip,
            'early_stopping_patience': self.early_stopping_patience,
            'data_dir': self.data_dir,
            'train_data_path': self.train_data_path,
            'val_data_path': self.val_data_path,
            'test_data_path': self.test_data_path,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'checkpoint_path': self.checkpoint_path,
            'embeddings_path': self.embeddings_path,
            'max_eval_samples': self.max_eval_samples,
            'top_k': self.top_k,
        }

    def save_yaml(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def load_config(path: str) -> Config:
    """Load from YAML or JSON file."""
    if path.endswith('.yaml') or path.endswith('.yml'):
        return Config.from_yaml(path)
    elif path.endswith('.json'):
        return Config.from_json(path)
    else:
        raise ValueError(f"Unsupported config format: {path}")


DEFAULT_CONFIGS = {
    'ngram': {
        'model_type': 'ngram',
        'n': 3,
        'smoothing': 'kneser_ney',
        'alpha': 0.75,
    },
    'fnn': {
        'model_type': 'fnn',
        'embedding_dim': 300,
        'hidden_size': 512,
        'context_size': 10,
        'num_layers': 2,
        'dropout': 0.3,
    },
    'rnn': {
        'model_type': 'rnn',
        'embedding_dim': 300,
        'hidden_size': 512,
        'num_layers': 2,
        'dropout': 0.3,
    },
    'lstm': {
        'model_type': 'lstm',
        'embedding_dim': 300,
        'hidden_size': 512,
        'num_layers': 2,
        'dropout': 0.3,
    },
    'gru': {
        'model_type': 'gru',
        'embedding_dim': 300,
        'hidden_size': 512,
        'num_layers': 2,
        'dropout': 0.3,
    },
}


def get_default_config(model_type: str) -> Config:
    if model_type not in DEFAULT_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")

    return Config.from_dict(DEFAULT_CONFIGS[model_type])
