#!/usr/bin/env python3
"""
Prediction/Inference Script for Next Word Prediction Models

Provides interactive and batch prediction capabilities.
"""

import os
import sys
import argparse
import json
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import NgramModel, FeedforwardLM, RNNModel, LSTMModel, GRUModel
from utils.data_utils import load_processed_data


def load_model(path: str):
    """Load model from checkpoint."""
    import torch

    if path.endswith('.pkl'):
        return NgramModel.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')
        model_type = checkpoint.get('model_type', 'lstm').lower()

        if model_type == 'lstm':
            return LSTMModel.load(path)
        elif model_type == 'gru':
            return GRUModel.load(path)
        elif model_type == 'rnn':
            return RNNModel.load(path)
        elif model_type == 'fnn':
            return FeedforwardLM.load(path)
        else:
            return LSTMModel.load(path)


class Predictor:
    """Next word prediction interface."""

    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab

        # Build word to index mapping
        if hasattr(vocab, 'word2idx'):
            self.word2idx = vocab.word2idx
        else:
            self.word2idx = {word: idx for idx, word in vocab.idx2word.items()}

        self.idx2word = vocab.idx2word if hasattr(vocab, 'idx2word') else {}

    def encode_text(self, text: str) -> List[int]:
        """Convert text to token indices."""
        words = text.lower().split()
        indices = []

        unk_idx = self.word2idx.get('<UNK>', self.word2idx.get('<unk>', 1))

        for word in words:
            idx = self.word2idx.get(word, unk_idx)
            indices.append(idx)

        return indices

    def decode_indices(self, indices: List[int]) -> List[str]:
        """Convert token indices to words."""
        return [self.idx2word.get(idx, '<UNK>') for idx in indices]

    def predict(self, context: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict next word given context.

        Args:
            context: Input text (space-separated words)
            top_k: Number of top predictions to return

        Returns:
            List of (word, probability) tuples
        """
        indices = self.encode_text(context)

        if not indices:
            return []

        predictions = self.model.predict_next(indices, top_k=top_k)

        # Convert indices to words
        results = []
        for idx, prob in predictions:
            word = self.idx2word.get(idx, '<UNK>')
            results.append((word, prob))

        return results

    def complete_sentence(self, context: str, max_words: int = 10,
                         stop_tokens: Optional[List[str]] = None) -> str:
        """
        Complete a sentence by repeatedly predicting next word.

        Args:
            context: Starting text
            max_words: Maximum words to generate
            stop_tokens: Tokens to stop generation (default: ['.', '!', '?', '<EOS>'])

        Returns:
            Completed sentence
        """
        if stop_tokens is None:
            stop_tokens = ['.', '!', '?', '<EOS>', '<eos>']

        words = context.split()

        for _ in range(max_words):
            predictions = self.predict(' '.join(words), top_k=1)

            if not predictions:
                break

            next_word = predictions[0][0]
            words.append(next_word)

            if next_word in stop_tokens:
                break

        return ' '.join(words)

    def get_probability(self, context: str, target: str) -> float:
        """
        Get probability of a specific word given context.

        Args:
            context: Input text
            target: Target word to get probability for

        Returns:
            Probability of target word
        """
        predictions = self.predict(context, top_k=100)

        for word, prob in predictions:
            if word.lower() == target.lower():
                return prob

        return 0.0


def interactive_mode(predictor: Predictor, top_k: int = 5):
    """Run interactive prediction mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE NEXT WORD PREDICTION")
    print("=" * 60)
    print("Enter text and get predictions for the next word.")
    print("Commands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'complete: <text>' - Auto-complete a sentence")
    print("  'prob: <context> | <word>' - Get probability of specific word")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        # Complete sentence command
        if user_input.lower().startswith('complete:'):
            context = user_input[9:].strip()
            if context:
                result = predictor.complete_sentence(context, max_words=15)
                print(f"Completion: {result}\n")
            else:
                print("Please provide context after 'complete:'\n")
            continue

        # Probability command
        if user_input.lower().startswith('prob:'):
            parts = user_input[5:].split('|')
            if len(parts) == 2:
                context = parts[0].strip()
                target = parts[1].strip()
                prob = predictor.get_probability(context, target)
                print(f"P({target} | {context}) = {prob:.6f}\n")
            else:
                print("Format: prob: <context> | <word>\n")
            continue

        # Default: predict next word
        predictions = predictor.predict(user_input, top_k=top_k)

        if predictions:
            print(f"\nTop {len(predictions)} predictions:")
            for i, (word, prob) in enumerate(predictions, 1):
                print(f"  {i}. {word:20s} (p={prob:.4f})")
            print()
        else:
            print("Could not make predictions for this input.\n")


def batch_predict(predictor: Predictor, input_file: str, output_file: str, top_k: int = 5):
    """
    Run batch prediction on a file.

    Args:
        predictor: Predictor instance
        input_file: Path to input file (one context per line)
        output_file: Path to output JSON file
    """
    results = []

    with open(input_file, 'r') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} inputs...")

    for line in lines:
        context = line.strip()
        if not context:
            continue

        predictions = predictor.predict(context, top_k=top_k)

        results.append({
            'context': context,
            'predictions': [
                {'word': word, 'probability': prob}
                for word, prob in predictions
            ]
        })

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Next Word Prediction Inference")

    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='processed_data',
                       help='Data directory (for vocabulary)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top predictions')
    parser.add_argument('--input', type=str,
                       help='Input file for batch prediction (one context per line)')
    parser.add_argument('--output', type=str,
                       help='Output file for batch prediction results')
    parser.add_argument('--context', type=str,
                       help='Single context for prediction (non-interactive)')
    parser.add_argument('--complete', action='store_true',
                       help='Complete the sentence instead of just predicting next word')

    args = parser.parse_args()

    # Load vocabulary
    vocab, _, _, _ = load_processed_data(args.data_dir)

    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    print(f"Model loaded: {model.get_model_info()}")

    # Create predictor
    predictor = Predictor(model, vocab)

    # Determine mode
    if args.input and args.output:
        # Batch mode
        batch_predict(predictor, args.input, args.output, args.top_k)
    elif args.context:
        # Single prediction mode
        if args.complete:
            result = predictor.complete_sentence(args.context, max_words=15)
            print(f"Completion: {result}")
        else:
            predictions = predictor.predict(args.context, top_k=args.top_k)
            print(f"\nContext: \"{args.context}\"")
            print(f"\nTop {len(predictions)} predictions:")
            for i, (word, prob) in enumerate(predictions, 1):
                print(f"  {i}. {word:20s} (p={prob:.4f})")
    else:
        # Interactive mode
        interactive_mode(predictor, args.top_k)


if __name__ == '__main__':
    main()
