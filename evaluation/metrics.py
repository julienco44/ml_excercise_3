# Evaluation metrics: accuracy, perplexity, MRR, etc.

import math
from typing import List, Dict, Tuple
import numpy as np
import torch
from tqdm import tqdm


def calculate_accuracy(predictions: List[int], targets: List[int]) -> float:
    """Exact match accuracy."""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(predictions) if predictions else 0.0


def calculate_topk_accuracy(predictions_list: List[List[Tuple[int, float]]],
                           targets: List[int], k: int = 5) -> float:
    """Check if target is in top-k predictions."""
    if len(predictions_list) != len(targets):
        raise ValueError("Predictions and targets must have the same length")

    correct = 0
    for preds, target in zip(predictions_list, targets):
        top_k_words = [p[0] for p in preds[:k]]
        if target in top_k_words:
            correct += 1
    return correct / len(targets) if targets else 0.0


def calculate_mrr(predictions_list: List[List[Tuple[int, float]]],
                  targets: List[int]) -> float:
    """Mean Reciprocal Rank - average of 1/rank for correct predictions."""
    reciprocal_ranks = []
    for preds, target in zip(predictions_list, targets):
        ranked_words = [p[0] for p in preds]
        try:
            rank = ranked_words.index(target) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def calculate_perplexity(model, sequences: List[List[int]],
                        batch_size: int = 64) -> float:
    return model.calculate_perplexity(sequences)


def evaluate_model(model, test_sequences: List[List[int]], vocab=None,
                   top_k: int = 10, max_samples: int = None,
                   show_progress: bool = True) -> Dict:
    """Run full evaluation and return dict of metrics."""
    predictions = []
    top_k_predictions = []
    targets = []

    # Get context size for the model
    if hasattr(model, 'context_size'):
        context_size = model.context_size
    elif hasattr(model, 'n'):
        context_size = model.n - 1
    else:
        context_size = 10  # Default

    sample_count = 0
    iterator = tqdm(test_sequences, desc="Evaluating") if show_progress else test_sequences

    for seq in iterator:
        if max_samples and sample_count >= max_samples:
            break

        # Skip very short sequences
        if len(seq) < 2:
            continue

        # Evaluate at each position
        for i in range(1, min(len(seq), 20)):  # Limit positions per sequence
            if max_samples and sample_count >= max_samples:
                break

            context = seq[:i]
            target = seq[i]

            # Skip special tokens as targets
            if target < 4:  # PAD, UNK, SOS, EOS
                continue

            # Get predictions
            try:
                top_k_preds = model.predict_next(context, top_k=top_k)

                if top_k_preds:
                    predictions.append(top_k_preds[0][0])
                    top_k_predictions.append(top_k_preds)
                    targets.append(target)
                    sample_count += 1
            except Exception as e:
                continue

    # Calculate metrics
    results = {
        'num_samples': len(predictions),
        'accuracy': calculate_accuracy(predictions, targets),
        'top_3_accuracy': calculate_topk_accuracy(top_k_predictions, targets, k=3),
        'top_5_accuracy': calculate_topk_accuracy(top_k_predictions, targets, k=5),
        'top_10_accuracy': calculate_topk_accuracy(top_k_predictions, targets, k=10),
        'mrr': calculate_mrr(top_k_predictions, targets),
    }

    # Calculate perplexity on subset
    perplexity_sequences = test_sequences[:1000] if max_samples else test_sequences
    try:
        results['perplexity'] = model.calculate_perplexity(perplexity_sequences)
    except Exception as e:
        results['perplexity'] = float('inf')

    return results


def print_evaluation_results(results: Dict, model_name: str = "Model"):
    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS: {model_name}")
    print('=' * 60)

    print(f"\nBasic Metrics:")
    print(f"  Samples evaluated: {results['num_samples']:,}")
    print(f"  Accuracy (exact match): {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Top-3 Accuracy: {results['top_3_accuracy']:.4f} ({results['top_3_accuracy']*100:.2f}%)")
    print(f"  Top-5 Accuracy: {results['top_5_accuracy']:.4f} ({results['top_5_accuracy']*100:.2f}%)")
    print(f"  Top-10 Accuracy: {results['top_10_accuracy']:.4f} ({results['top_10_accuracy']*100:.2f}%)")
    print(f"  Mean Reciprocal Rank: {results['mrr']:.4f}")
    print(f"  Perplexity: {results['perplexity']:.2f}")

    if 'avg_semantic_similarity' in results:
        print(f"\nSemantic Metrics:")
        print(f"  Avg Semantic Similarity: {results['avg_semantic_similarity']:.4f}")
        if 'semantic_accuracy_0.5' in results:
            print(f"  Semantic Accuracy (>0.5): {results['semantic_accuracy_0.5']:.4f}")
        if 'semantic_accuracy_0.7' in results:
            print(f"  Semantic Accuracy (>0.7): {results['semantic_accuracy_0.7']:.4f}")


def generate_sample_predictions(model, vocab, test_sequences: List[List[int]],
                               num_samples: int = 10, context_length: int = 5) -> List[Dict]:
    """Generate some example predictions for inspection."""
    samples = []
    np.random.seed(42)

    seq_indices = np.random.choice(len(test_sequences),
                                   min(num_samples * 3, len(test_sequences)),
                                   replace=False)

    for seq_idx in seq_indices:
        if len(samples) >= num_samples:
            break

        seq = test_sequences[seq_idx]
        if len(seq) < context_length + 2:
            continue

        # Select random position
        pos = np.random.randint(context_length, len(seq) - 1)

        context = seq[max(0, pos - context_length):pos + 1]
        target = seq[pos + 1]

        if target < 4:
            continue

        try:
            top_k_preds = model.predict_next(context, top_k=5)
        except:
            continue

        # Decode words
        context_words = vocab.decode(context) if hasattr(vocab, 'decode') else [vocab.idx2word.get(i, '<UNK>') for i in context]
        target_word = vocab.idx2word.get(target, '<UNK>') if hasattr(vocab, 'idx2word') else str(target)
        pred_words = [(vocab.idx2word.get(p[0], '<UNK>'), p[1]) for p in top_k_preds]

        samples.append({
            'context': ' '.join(context_words),
            'target': target_word,
            'predictions': pred_words,
            'is_correct': top_k_preds[0][0] == target if top_k_preds else False,
            'in_top_5': target in [p[0] for p in top_k_preds[:5]],
        })

    return samples


def print_samples(samples: List[Dict], title: str = "Sample Predictions"):
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print('=' * 60)

    for i, sample in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Context: \"{sample['context']}\"")
        print(f"Target: \"{sample['target']}\"")
        print(f"Predictions:")
        for j, (word, prob) in enumerate(sample['predictions'], 1):
            marker = "*" if word == sample['target'] else " "
            print(f"  {j}. {marker} {word} (p={prob:.4f})")
        status = "CORRECT" if sample['is_correct'] else ("In Top-5" if sample['in_top_5'] else "INCORRECT")
        print(f"Status: {status}")
