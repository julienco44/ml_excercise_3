#!/usr/bin/env python3
"""
K-Fold Cross-Validation Script for Next Word Prediction Models

Performs k-fold cross-validation and reports mean ± std for all metrics.
"""

import os
import sys
import argparse
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import NgramModel, FeedforwardLM, RNNModel, LSTMModel, GRUModel
from utils.config import Config, load_config, get_default_config
from utils.data_utils import load_processed_data, save_results
from evaluation.metrics import evaluate_model
from evaluation.statistical_tests import bootstrap_ci, cohens_d


MODEL_CLASSES = {
    'ngram': NgramModel,
    'fnn': FeedforwardLM,
    'rnn': RNNModel,
    'lstm': LSTMModel,
    'gru': GRUModel,
}


def create_model(config: Config, vocab_size: int):
    """Create model based on configuration."""
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


def cross_validate(config: Config, all_sequences: List[List[int]],
                   vocab, k_folds: int = 5,
                   max_eval_samples: int = 5000) -> Dict:
    """
    Perform k-fold cross-validation.

    Args:
        config: Model configuration
        all_sequences: All training sequences (will be split into folds)
        vocab: Vocabulary object
        k_folds: Number of folds
        max_eval_samples: Maximum samples for evaluation per fold

    Returns:
        Dictionary with CV results including mean and std for each metric
    """
    print(f"\n{'=' * 60}")
    print(f"K-FOLD CROSS-VALIDATION: {config.experiment_name}")
    print(f"Model Type: {config.model_type}")
    print(f"Folds: {k_folds}")
    print(f"{'=' * 60}")

    vocab_size = len(vocab)
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Convert to numpy for indexing
    sequences_array = np.array(all_sequences, dtype=object)

    fold_results = {
        'accuracy': [],
        'top_3_accuracy': [],
        'top_5_accuracy': [],
        'top_10_accuracy': [],
        'mrr': [],
        'perplexity': [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(sequences_array)):
        print(f"\n--- Fold {fold_idx + 1}/{k_folds} ---")

        # Split data
        train_fold = list(sequences_array[train_idx])
        val_fold = list(sequences_array[val_idx])

        print(f"Train size: {len(train_fold)}, Val size: {len(val_fold)}")

        # Create fresh model
        model = create_model(config, vocab_size)

        # Train
        if config.model_type.lower() == 'ngram':
            model.train_model(train_fold, val_fold)
        else:
            model.train_model(
                train_fold, val_fold,
                epochs=config.num_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                grad_clip=config.grad_clip,
                early_stopping_patience=config.early_stopping_patience
            )

        # Evaluate
        results = evaluate_model(
            model, val_fold, vocab,
            top_k=config.top_k,
            max_samples=max_eval_samples,
            show_progress=False
        )

        # Store results
        fold_results['accuracy'].append(results['accuracy'])
        fold_results['top_3_accuracy'].append(results['top_3_accuracy'])
        fold_results['top_5_accuracy'].append(results['top_5_accuracy'])
        fold_results['top_10_accuracy'].append(results['top_10_accuracy'])
        fold_results['mrr'].append(results['mrr'])
        fold_results['perplexity'].append(results['perplexity'])

        print(f"Fold {fold_idx + 1} - Accuracy: {results['accuracy']:.4f}, "
              f"Perplexity: {results['perplexity']:.2f}")

    # Calculate statistics
    cv_results = {
        'model_name': config.experiment_name,
        'model_type': config.model_type,
        'k_folds': k_folds,
        'fold_results': fold_results,
    }

    for metric, values in fold_results.items():
        cv_results[f'{metric}_mean'] = np.mean(values)
        cv_results[f'{metric}_std'] = np.std(values)
        cv_results[f'{metric}_min'] = np.min(values)
        cv_results[f'{metric}_max'] = np.max(values)

        # Bootstrap confidence intervals
        try:
            ci_lower, ci_upper = bootstrap_ci(values, num_bootstrap=1000, confidence=0.95)
            cv_results[f'{metric}_ci_lower'] = ci_lower
            cv_results[f'{metric}_ci_upper'] = ci_upper
        except:
            pass

    return cv_results


def print_cv_results(results: Dict):
    """Pretty print cross-validation results."""
    print(f"\n{'=' * 60}")
    print(f"CROSS-VALIDATION RESULTS: {results['model_name']}")
    print(f"{'=' * 60}")

    metrics = ['accuracy', 'top_3_accuracy', 'top_5_accuracy', 'top_10_accuracy', 'mrr', 'perplexity']

    for metric in metrics:
        mean = results[f'{metric}_mean']
        std = results[f'{metric}_std']

        if metric == 'perplexity':
            print(f"{metric:20s}: {mean:8.2f} ± {std:.2f}")
        else:
            print(f"{metric:20s}: {mean:8.4f} ± {std:.4f} ({mean*100:.2f}% ± {std*100:.2f}%)")

        if f'{metric}_ci_lower' in results:
            ci_lower = results[f'{metric}_ci_lower']
            ci_upper = results[f'{metric}_ci_upper']
            if metric == 'perplexity':
                print(f"{'':20s}  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
            else:
                print(f"{'':20s}  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")


def compare_cv_results(results_list: List[Dict]) -> Dict:
    """
    Compare cross-validation results from multiple models.

    Args:
        results_list: List of CV results dictionaries

    Returns:
        Comparison statistics including effect sizes
    """
    comparison = {
        'models': [r['model_name'] for r in results_list],
        'metrics': {},
    }

    metrics = ['accuracy', 'top_5_accuracy', 'perplexity', 'mrr']

    for metric in metrics:
        comparison['metrics'][metric] = {
            'best_model': None,
            'best_value': None,
            'rankings': [],
        }

        # Sort models by metric (ascending for perplexity, descending for others)
        if metric == 'perplexity':
            sorted_results = sorted(results_list, key=lambda x: x[f'{metric}_mean'])
        else:
            sorted_results = sorted(results_list, key=lambda x: x[f'{metric}_mean'], reverse=True)

        comparison['metrics'][metric]['best_model'] = sorted_results[0]['model_name']
        comparison['metrics'][metric]['best_value'] = sorted_results[0][f'{metric}_mean']
        comparison['metrics'][metric]['rankings'] = [
            {'model': r['model_name'], 'mean': r[f'{metric}_mean'], 'std': r[f'{metric}_std']}
            for r in sorted_results
        ]

        # Calculate effect sizes between consecutive models
        if len(results_list) > 1:
            effects = []
            for i in range(len(sorted_results) - 1):
                d = cohens_d(
                    sorted_results[i]['fold_results'][metric],
                    sorted_results[i+1]['fold_results'][metric]
                )
                effects.append({
                    'comparison': f"{sorted_results[i]['model_name']} vs {sorted_results[i+1]['model_name']}",
                    'cohens_d': d,
                })
            comparison['metrics'][metric]['effect_sizes'] = effects

    return comparison


def main():
    parser = argparse.ArgumentParser(description="K-Fold Cross-Validation for NWP Models")

    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--model', type=str, choices=['ngram', 'fnn', 'rnn', 'lstm', 'gru'],
                       help='Model type')
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--k-folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--data-dir', type=str, default='processed_data', help='Data directory')
    parser.add_argument('--output', type=str, default='experiments/results', help='Output directory')
    parser.add_argument('--max-samples', type=int, default=5000,
                       help='Maximum samples for evaluation per fold')

    # Model hyperparameters (optional overrides)
    parser.add_argument('--embedding-dim', type=int)
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--num-layers', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--n', type=int, help='N-gram order')
    parser.add_argument('--smoothing', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--learning-rate', type=float)

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = load_config(args.config)
    elif args.model:
        config = get_default_config(args.model)
    else:
        parser.error("Either --config or --model is required")

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
    if args.n:
        config.n = args.n
    if args.smoothing:
        config.smoothing = args.smoothing
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate

    config.data_dir = args.data_dir

    # Set experiment name if not provided
    if not args.name and not args.config:
        config.experiment_name = f"{config.model_type}_cv_{args.k_folds}fold"

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load data
    vocab, train_sequences, val_sequences, test_sequences = load_processed_data(args.data_dir)

    # Combine train and val for CV (will re-split in each fold)
    all_sequences = train_sequences + val_sequences
    print(f"Total sequences for CV: {len(all_sequences)}")

    # Run cross-validation
    cv_results = cross_validate(
        config, all_sequences, vocab,
        k_folds=args.k_folds,
        max_eval_samples=args.max_samples
    )

    # Print results
    print_cv_results(cv_results)

    # Save results
    cv_results['config'] = config.to_dict()
    cv_results['timestamp'] = datetime.now().isoformat()

    output_path = os.path.join(args.output, f"{config.experiment_name}_cv_results.json")
    save_results(cv_results, output_path)

    print(f"\nCross-validation complete!")
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
