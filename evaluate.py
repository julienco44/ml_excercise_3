#!/usr/bin/env python3
"""
Evaluation Script for Next Word Prediction Models

Evaluates trained models and generates comparison reports.
"""

import os
import sys
import argparse
import json
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import NgramModel, FeedforwardLM, RNNModel, LSTMModel, GRUModel
from utils.data_utils import load_processed_data, save_results, download_glove_embeddings
from utils.visualization import plot_model_comparison, plot_all_metrics, plot_perplexity_comparison
from evaluation.metrics import evaluate_model, print_evaluation_results, generate_sample_predictions, print_samples
from evaluation.semantic_similarity import SemanticEvaluator


def load_model(path: str):
    import torch
    from models import FeedforwardLM, LSTMModel, GRUModel, RNNModel, NgramModel

    # N-gram models
    if path.endswith(".pkl"):
        return NgramModel.load(path)

    checkpoint = torch.load(path, map_location="cpu")

    # Prefer filename-based detection (robust)
    filename = os.path.basename(path).lower()

    if "fnn" in filename:
        return FeedforwardLM.load(path)
    elif "lstm" in filename:
        return LSTMModel.load(path)
    elif "gru" in filename:
        return GRUModel.load(path)
    elif "rnn" in filename:
        return RNNModel.load(path)

    # Fallback: try model_type inside checkpoint
    model_type = checkpoint.get("model_type", "").lower()

    if model_type == "fnn":
        return FeedforwardLM.load(path)
    elif model_type == "lstm":
        return LSTMModel.load(path)
    elif model_type == "gru":
        return GRUModel.load(path)
    elif model_type == "rnn":
        return RNNModel.load(path)

    raise ValueError(f"Unknown model type for checkpoint: {path}")



def evaluate_single_model(model_path: str, test_data: List[List[int]], vocab,
                         semantic_evaluator=None, max_samples: int = 10000,
                         num_sample_predictions: int = 10) -> Dict:
    """Evaluate a single model."""
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path)

    model_name = os.path.splitext(os.path.basename(model_path))[0]

    # Basic evaluation
    results = evaluate_model(
        model, test_data, vocab,
        top_k=10,
        max_samples=max_samples
    )
    results['model_name'] = model_name
    results['model_path'] = model_path
    results['model_info'] = model.get_model_info()

    # Semantic evaluation if available
    if semantic_evaluator and semantic_evaluator.embeddings is not None:
        # Get predictions for semantic evaluation
        predictions = []
        top_k_predictions = []
        targets = []

        for seq in test_data[:1000]:
            if len(seq) < 2:
                continue
            for i in range(1, min(len(seq), 10)):
                context = seq[:i]
                target = seq[i]
                if target < 4:
                    continue

                try:
                    preds = model.predict_next(context, top_k=10)
                    if preds:
                        predictions.append(preds[0][0])
                        top_k_predictions.append(preds)
                        targets.append(target)
                except:
                    continue

                if len(predictions) >= 5000:
                    break
            if len(predictions) >= 5000:
                break

        semantic_results = semantic_evaluator.evaluate_semantic(
            predictions, top_k_predictions, targets
        )
        results.update(semantic_results)

    # Generate sample predictions
    if num_sample_predictions > 0:
        samples = generate_sample_predictions(
            model, vocab, test_data,
            num_samples=num_sample_predictions
        )
        results['sample_predictions'] = samples

    return results


def compare_models(results: List[Dict]) -> Dict:
    """Generate comparison statistics."""
    comparison = {
        'models': [r['model_name'] for r in results],
        'best_accuracy': max(results, key=lambda x: x['accuracy'])['model_name'],
        'best_top5_accuracy': max(results, key=lambda x: x['top_5_accuracy'])['model_name'],
        'best_perplexity': min(results, key=lambda x: x['perplexity'])['model_name'],
        'best_mrr': max(results, key=lambda x: x['mrr'])['model_name'],
    }

    if 'avg_semantic_similarity' in results[0]:
        comparison['best_semantic'] = max(results, key=lambda x: x.get('avg_semantic_similarity', 0))['model_name']

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate Next Word Prediction Models")

    parser.add_argument('--models', nargs='+', required=True,
                       help='Paths to model checkpoints')
    parser.add_argument('--data-dir', type=str, default='processed_data',
                       help='Data directory')
    parser.add_argument('--output', type=str, default='experiments/results',
                       help='Output directory for results')
    parser.add_argument('--max-samples', type=int, default=10000,
                       help='Maximum samples for evaluation')
    parser.add_argument('--num-samples', type=int, default=15,
                       help='Number of sample predictions to show')
    parser.add_argument('--embeddings', type=str, default='embeddings/glove.6B.100d.txt',
                       help='Path to embeddings for semantic evaluation')
    parser.add_argument('--no-semantic', action='store_true',
                       help='Skip semantic evaluation')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load data
    vocab, train_sequences, val_sequences, test_sequences = load_processed_data(args.data_dir)

    # Setup semantic evaluator
    semantic_evaluator = None
    if not args.no_semantic:
        if not os.path.exists(args.embeddings):
            embeddings_path = download_glove_embeddings(dim=100)
        else:
            embeddings_path = args.embeddings

        if embeddings_path:
            semantic_evaluator = SemanticEvaluator(vocab, embeddings_path, embedding_dim=100)

    # Evaluate each model
    all_results = []
    for model_path in args.models:
        print("\n" + "=" * 60)
        print(f"Evaluating: {model_path}")
        print("=" * 60)

        results = evaluate_single_model(
            model_path, test_sequences, vocab,
            semantic_evaluator=semantic_evaluator,
            max_samples=args.max_samples,
            num_sample_predictions=args.num_samples
        )
        all_results.append(results)

        print_evaluation_results(results, results['model_name'])

        if 'sample_predictions' in results and results['sample_predictions']:
            print_samples(results['sample_predictions'][:5],
                         title=f"Sample Predictions - {results['model_name']}")

    # Compare models
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        comparison = compare_models(all_results)

        print(f"\nBest by Accuracy: {comparison['best_accuracy']}")
        print(f"Best by Top-5 Accuracy: {comparison['best_top5_accuracy']}")
        print(f"Best by Perplexity: {comparison['best_perplexity']}")
        print(f"Best by MRR: {comparison['best_mrr']}")
        if 'best_semantic' in comparison:
            print(f"Best by Semantic Similarity: {comparison['best_semantic']}")

        # Generate comparison plots
        plot_model_comparison(all_results, metric='accuracy',
                            save_path=os.path.join(args.output, 'comparison_accuracy.png'))
        plot_perplexity_comparison(all_results,
                                  save_path=os.path.join(args.output, 'comparison_perplexity.png'))
        plot_all_metrics(all_results,
                        save_path=os.path.join(args.output, 'comparison_all_metrics.png'))

    # Save all results
    output_data = {
        'results': [{k: v for k, v in r.items() if k != 'sample_predictions'} for r in all_results],
        'comparison': compare_models(all_results) if len(all_results) > 1 else {},
        'sample_predictions': {r['model_name']: r.get('sample_predictions', []) for r in all_results},
    }

    results_path = os.path.join(args.output, 'evaluation_results.json')
    save_results(output_data, results_path)

    print(f"\nEvaluation complete!")
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
