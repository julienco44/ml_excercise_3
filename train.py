#!/usr/bin/env python3
# Training script for next word prediction models

import os
import sys
import argparse
import random
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import NgramModel, FeedforwardLM, RNNModel, LSTMModel, GRUModel
from utils.config import Config, load_config, get_default_config
from utils.data_utils import load_processed_data, save_results
from utils.visualization import plot_training_history
from evaluation.metrics import evaluate_model, print_evaluation_results


MODEL_CLASSES = {
    'ngram': NgramModel,
    'fnn': FeedforwardLM,
    'rnn': RNNModel,
    'lstm': LSTMModel,
    'gru': GRUModel,
}


def create_model(config: Config, vocab_size: int):
    model_type = config.model_type.lower()

    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unknown model type: {model_type}")

    model_class = MODEL_CLASSES[model_type]

    if model_type == 'ngram':
        return model_class(
            vocab_size=vocab_size,
            n=config.n,
            smoothing=config.smoothing,
            alpha=config.alpha
        )
    elif model_type == 'fnn':
        return model_class(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            context_size=config.context_size,
            dropout=config.dropout,
            num_layers=config.num_layers
        )
    else:  # rnn, lstm, gru
        return model_class(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout
        )


def train(config: Config, sample_fraction: float = None):
    print("=" * 60)
    print(f"TRAINING: {config.experiment_name}")
    print(f"Model Type: {config.model_type}")
    print("=" * 60)

    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Load data
    vocab, train_sequences, val_sequences, test_sequences = load_processed_data(config.data_dir)
    vocab_size = len(vocab)
    config.vocab_size = vocab_size

    # Sample training data if requested (to speed up experiments)
    if sample_fraction and sample_fraction < 1.0:
        random.seed(42)
        n_samples = int(len(train_sequences) * sample_fraction)
        train_sequences = random.sample(train_sequences, n_samples)
        print(f"\nSampled {n_samples} training sequences ({sample_fraction*100:.0f}% of data)")

    print(f"\nVocabulary size: {vocab_size}")
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Val sequences: {len(val_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")

    # Create model
    model = create_model(config, vocab_size)
    print(f"\nModel created: {model.get_model_info()}")

    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    if config.model_type.lower() == 'ngram':
        history = model.train_model(train_sequences, val_sequences)
    else:
        history = model.train_model(
            train_sequences, val_sequences,
            epochs=config.num_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            grad_clip=config.grad_clip,
            early_stopping_patience=config.early_stopping_patience
        )

    # Save model
    model.save(config.checkpoint_path)

    # Plot training history
    if 'train_loss' in history:
        plot_path = os.path.join(config.output_dir, f"{config.experiment_name}_training.png")
        plot_training_history(history, save_path=plot_path, title=config.experiment_name)

    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("VALIDATION EVALUATION")
    print("=" * 60)

    val_results = evaluate_model(
        model, val_sequences, vocab,
        top_k=config.top_k,
        max_samples=config.max_eval_samples
    )
    val_results['model_name'] = config.experiment_name
    print_evaluation_results(val_results, f"{config.experiment_name} (Validation)")

    # Save results
    results = {
        'config': config.to_dict(),
        'training_history': history,
        'validation_results': val_results,
        'timestamp': datetime.now().isoformat(),
    }

    results_path = os.path.join(config.output_dir, f"{config.experiment_name}_results.json")
    save_results(results, results_path)

    print(f"\nTraining complete!")
    print(f"Model saved to: {config.checkpoint_path}")
    print(f"Results saved to: {results_path}")

    return model, results


def main():
    parser = argparse.ArgumentParser(description="Train Next Word Prediction Model")

    parser.add_argument('--config', type=str, help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--model', type=str, choices=['ngram', 'fnn', 'rnn', 'lstm', 'gru'],
                       help='Model type (overrides config)')
    parser.add_argument('--name', type=str, help='Experiment name (overrides config)')

    # Model hyperparameters
    parser.add_argument('--embedding-dim', type=int, help='Embedding dimension')
    parser.add_argument('--hidden-size', type=int, help='Hidden layer size')
    parser.add_argument('--num-layers', type=int, help='Number of layers')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--context-size', type=int, help='Context size for FNN')

    # N-gram specific
    parser.add_argument('--n', type=int, help='N-gram order')
    parser.add_argument('--smoothing', type=str, choices=['laplace', 'kneser_ney', 'interpolation'],
                       help='Smoothing method for n-gram')
    parser.add_argument('--alpha', type=float, help='Smoothing parameter')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--grad-clip', type=float, help='Gradient clipping')

    # Paths
    parser.add_argument('--data-dir', type=str, default='processed_data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='experiments/results', help='Output directory')
    parser.add_argument('--checkpoint-dir', type=str, default='experiments/checkpoints', help='Checkpoint directory')

    # Data sampling (for faster experiments)
    parser.add_argument('--sample-fraction', type=float, default=None,
                       help='Fraction of training data to use (0.0-1.0)')

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = load_config(args.config)
    elif args.model:
        config = get_default_config(args.model)
    else:
        config = Config()

    # Override with command-line arguments
    if args.model:
        config.model_type = args.model
    if args.name:
        config.experiment_name = args.name
    if args.embedding_dim:
        config.embedding_dim = args.embedding_dim
    if args.hidden_size:
        config.hidden_size = args.hidden_size
    if args.num_layers:
        config.num_layers = args.num_layers
    if args.dropout:
        config.dropout = args.dropout
    if args.context_size:
        config.context_size = args.context_size
    if args.n:
        config.n = args.n
    if args.smoothing:
        config.smoothing = args.smoothing
    if args.alpha:
        config.alpha = args.alpha
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.grad_clip:
        config.grad_clip = args.grad_clip
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir

    # Set experiment name if not provided
    if not args.name and not args.config:
        config.experiment_name = f"{config.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Update derived paths
    config.__post_init__()

    # Train
    train(config, sample_fraction=args.sample_fraction)


if __name__ == '__main__':
    main()
