"""
Evaluation Metrics for Next Word Prediction

Includes:
- Accuracy (exact match)
- Top-k Accuracy
- Perplexity
- Semantic Similarity (using word embeddings)
- Mean Reciprocal Rank (MRR)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math


class Evaluator:
    """Comprehensive evaluation for next-word prediction models."""

    def __init__(self, vocab, embeddings: Optional[np.ndarray] = None):
        """
        Initialize evaluator.

        Args:
            vocab: Vocabulary object with word2idx and idx2word mappings
            embeddings: Optional word embeddings matrix (vocab_size x embedding_dim)
        """
        self.vocab = vocab
        self.embeddings = embeddings
        self.embedding_dim = embeddings.shape[1] if embeddings is not None else None

        # Precompute embedding norms for similarity calculations
        if embeddings is not None:
            self.embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            self.embedding_norms = np.where(self.embedding_norms == 0, 1, self.embedding_norms)
            self.normalized_embeddings = embeddings / self.embedding_norms

    def accuracy(self, predictions: List[int], targets: List[int]) -> float:
        """
        Calculate exact match accuracy.

        Args:
            predictions: List of predicted word indices
            targets: List of target word indices

        Returns:
            Accuracy as float between 0 and 1
        """
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have the same length")

        correct = sum(1 for p, t in zip(predictions, targets) if p == t)
        return correct / len(predictions) if predictions else 0.0

    def top_k_accuracy(self, predictions_list: List[List[Tuple[int, float]]],
                       targets: List[int], k: int = 5) -> float:
        """
        Calculate top-k accuracy.

        Args:
            predictions_list: List of top-k predictions for each sample
                              Each element is a list of (word_idx, probability) tuples
            targets: List of target word indices
            k: Number of top predictions to consider

        Returns:
            Top-k accuracy as float between 0 and 1
        """
        if len(predictions_list) != len(targets):
            raise ValueError("Predictions and targets must have the same length")

        correct = 0
        for preds, target in zip(predictions_list, targets):
            top_k_words = [p[0] for p in preds[:k]]
            if target in top_k_words:
                correct += 1

        return correct / len(targets) if targets else 0.0

    def mean_reciprocal_rank(self, predictions_list: List[List[Tuple[int, float]]],
                             targets: List[int]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        Args:
            predictions_list: List of ranked predictions for each sample
            targets: List of target word indices

        Returns:
            MRR score between 0 and 1
        """
        reciprocal_ranks = []

        for preds, target in zip(predictions_list, targets):
            ranked_words = [p[0] for p in preds]
            try:
                rank = ranked_words.index(target) + 1
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    def cosine_similarity(self, word1_idx: int, word2_idx: int) -> float:
        """
        Calculate cosine similarity between two words using embeddings.

        Args:
            word1_idx: Index of first word
            word2_idx: Index of second word

        Returns:
            Cosine similarity between -1 and 1
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded")

        if word1_idx >= len(self.embeddings) or word2_idx >= len(self.embeddings):
            return 0.0

        return float(np.dot(self.normalized_embeddings[word1_idx],
                           self.normalized_embeddings[word2_idx]))

    def semantic_accuracy(self, predictions: List[int], targets: List[int],
                          threshold: float = 0.7) -> float:
        """
        Calculate semantic accuracy using embedding similarity.
        A prediction is "semantically correct" if its cosine similarity
        with the target exceeds the threshold.

        Args:
            predictions: List of predicted word indices
            targets: List of target word indices
            threshold: Similarity threshold for considering a prediction correct

        Returns:
            Semantic accuracy as float between 0 and 1
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded for semantic evaluation")

        correct = 0
        valid = 0

        for pred, target in zip(predictions, targets):
            # Skip special tokens
            if pred < 4 or target < 4:  # PAD, UNK, SOS, EOS
                continue

            similarity = self.cosine_similarity(pred, target)
            if similarity >= threshold or pred == target:
                correct += 1
            valid += 1

        return correct / valid if valid > 0 else 0.0

    def average_semantic_similarity(self, predictions: List[int],
                                    targets: List[int]) -> float:
        """
        Calculate average semantic similarity between predictions and targets.

        Args:
            predictions: List of predicted word indices
            targets: List of target word indices

        Returns:
            Average cosine similarity
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded")

        similarities = []
        for pred, target in zip(predictions, targets):
            if pred < 4 or target < 4:
                continue
            sim = self.cosine_similarity(pred, target)
            similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

    def top_k_semantic_accuracy(self, predictions_list: List[List[Tuple[int, float]]],
                                targets: List[int], k: int = 5,
                                threshold: float = 0.7) -> float:
        """
        Calculate top-k semantic accuracy.
        A prediction is correct if any of the top-k predictions is semantically
        similar to the target (or exact match).

        Args:
            predictions_list: List of top-k predictions for each sample
            targets: List of target word indices
            k: Number of top predictions to consider
            threshold: Similarity threshold

        Returns:
            Top-k semantic accuracy
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded")

        correct = 0
        valid = 0

        for preds, target in zip(predictions_list, targets):
            if target < 4:
                continue

            top_k_words = [p[0] for p in preds[:k]]

            # Check exact match first
            if target in top_k_words:
                correct += 1
                valid += 1
                continue

            # Check semantic similarity
            found_similar = False
            for word_idx in top_k_words:
                if word_idx >= 4:  # Skip special tokens
                    similarity = self.cosine_similarity(word_idx, target)
                    if similarity >= threshold:
                        found_similar = True
                        break

            if found_similar:
                correct += 1
            valid += 1

        return correct / valid if valid > 0 else 0.0

    def evaluate_model(self, model, test_sequences: List[List[int]],
                       top_k: int = 10, max_samples: int = None) -> Dict:
        """
        Comprehensive evaluation of an n-gram model.

        Args:
            model: NGramModel instance
            test_sequences: List of test sequences
            top_k: Number of top predictions to generate
            max_samples: Maximum number of samples to evaluate (for speed)

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"Evaluating model on {len(test_sequences)} sequences...")

        predictions = []
        top_k_predictions = []
        targets = []

        n = model.n
        sample_count = 0

        for seq in test_sequences:
            if max_samples and sample_count >= max_samples:
                break

            padded_seq = [0] * (n - 1) + seq

            for i in range(len(padded_seq) - n + 1):
                if max_samples and sample_count >= max_samples:
                    break

                context = list(padded_seq[i:i + n - 1])
                target = padded_seq[i + n - 1]

                # Skip predicting special tokens
                if target < 4:  # PAD, UNK, SOS, EOS
                    continue

                # Get predictions
                top_k_preds = model.predict_next_fast(context, top_k=top_k)

                if top_k_preds:
                    predictions.append(top_k_preds[0][0])  # Top-1 prediction
                    top_k_predictions.append(top_k_preds)
                    targets.append(target)
                    sample_count += 1

        print(f"  Evaluated {len(predictions)} predictions")

        # Calculate metrics
        results = {
            'num_samples': len(predictions),
            'accuracy': self.accuracy(predictions, targets),
            'top_3_accuracy': self.top_k_accuracy(top_k_predictions, targets, k=3),
            'top_5_accuracy': self.top_k_accuracy(top_k_predictions, targets, k=5),
            'top_10_accuracy': self.top_k_accuracy(top_k_predictions, targets, k=10),
            'mrr': self.mean_reciprocal_rank(top_k_predictions, targets),
            'perplexity': model.calculate_perplexity(test_sequences[:1000] if max_samples else test_sequences),
        }

        # Add semantic metrics if embeddings available
        if self.embeddings is not None:
            results['avg_semantic_similarity'] = self.average_semantic_similarity(predictions, targets)
            results['semantic_accuracy_0.5'] = self.semantic_accuracy(predictions, targets, threshold=0.5)
            results['semantic_accuracy_0.7'] = self.semantic_accuracy(predictions, targets, threshold=0.7)
            results['top_5_semantic_accuracy'] = self.top_k_semantic_accuracy(
                top_k_predictions, targets, k=5, threshold=0.5)

        return results

    def generate_samples(self, model, vocab, test_sequences: List[List[int]],
                         num_samples: int = 10, context_length: int = 5) -> List[Dict]:
        """
        Generate sample predictions for manual inspection.

        Args:
            model: NGramModel instance
            vocab: Vocabulary object
            test_sequences: List of test sequences
            num_samples: Number of samples to generate
            context_length: Number of context words to show

        Returns:
            List of sample dictionaries with context, target, and predictions
        """
        samples = []
        n = model.n
        np.random.seed(42)  # For reproducibility

        # Select random sequences
        seq_indices = np.random.choice(len(test_sequences),
                                       min(num_samples * 2, len(test_sequences)),
                                       replace=False)

        for seq_idx in seq_indices:
            if len(samples) >= num_samples:
                break

            seq = test_sequences[seq_idx]
            if len(seq) < n + context_length:
                continue

            # Select random position in sequence
            pos = np.random.randint(n - 1, len(seq) - 1)

            context_start = max(0, pos - context_length + 1)
            context = seq[context_start:pos + 1]
            target = seq[pos + 1]

            # Skip special tokens as targets
            if target < 4:
                continue

            # Get predictions
            top_k_preds = model.predict_next_fast(context, top_k=5)

            # Decode words
            context_words = vocab.decode(context)
            target_word = vocab.idx2word.get(target, "<UNK>")
            pred_words = [(vocab.idx2word.get(p[0], "<UNK>"), p[1]) for p in top_k_preds]

            samples.append({
                'context': ' '.join(context_words),
                'target': target_word,
                'predictions': pred_words,
                'is_correct': top_k_preds[0][0] == target if top_k_preds else False,
                'in_top_5': target in [p[0] for p in top_k_preds[:5]],
            })

        return samples


def print_evaluation_results(results: Dict, model_name: str = "Model"):
    """Pretty print evaluation results."""
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
        print(f"  Semantic Accuracy (>0.5): {results['semantic_accuracy_0.5']:.4f} ({results['semantic_accuracy_0.5']*100:.2f}%)")
        print(f"  Semantic Accuracy (>0.7): {results['semantic_accuracy_0.7']:.4f} ({results['semantic_accuracy_0.7']*100:.2f}%)")
        print(f"  Top-5 Semantic Accuracy: {results['top_5_semantic_accuracy']:.4f} ({results['top_5_semantic_accuracy']*100:.2f}%)")


def print_samples(samples: List[Dict], title: str = "Sample Predictions"):
    """Pretty print sample predictions for manual assessment."""
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
