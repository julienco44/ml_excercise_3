# Tools for manual/qualitative evaluation of predictions

import os
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple


class QualitativeEvaluator:
    """
    Generate samples for manual rating.
    Rating scale: 0=invalid, 1=poor, 2=acceptable, 3=excellent
    """

    def __init__(self, model, vocab, random_seed: int = 42):
        self.model = model
        self.vocab = vocab
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Build word mappings
        if hasattr(vocab, 'idx2word'):
            self.idx2word = vocab.idx2word
        else:
            self.idx2word = {}

    def decode_sequence(self, indices: List[int]) -> str:
        words = [self.idx2word.get(idx, '<UNK>') for idx in indices]
        return ' '.join(words)

    def generate_samples(self, test_sequences: List[List[int]],
                        num_samples: int = 100,
                        context_length: int = 8,
                        stratify_by_confidence: bool = True) -> pd.DataFrame:
        all_samples = []

        # Generate more samples than needed for stratification
        target_raw_samples = num_samples * 5 if stratify_by_confidence else num_samples * 2

        seq_indices = np.random.permutation(len(test_sequences))

        for seq_idx in seq_indices:
            if len(all_samples) >= target_raw_samples:
                break

            seq = test_sequences[seq_idx]

            # Need at least context + target
            if len(seq) < context_length + 2:
                continue

            # Sample position within sequence
            pos = random.randint(context_length, len(seq) - 2)

            context = seq[max(0, pos - context_length):pos + 1]
            target = seq[pos + 1]

            # Skip special tokens as targets
            if target < 4:
                continue

            try:
                top_k_preds = self.model.predict_next(context, top_k=10)
            except:
                continue

            if not top_k_preds:
                continue

            predicted_idx, confidence = top_k_preds[0]
            is_correct = predicted_idx == target

            # Decode
            context_text = self.decode_sequence(context)
            target_word = self.idx2word.get(target, '<UNK>')
            predicted_word = self.idx2word.get(predicted_idx, '<UNK>')
            top_5_words = [self.idx2word.get(p[0], '<UNK>') for p in top_k_preds[:5]]

            all_samples.append({
                'sample_id': len(all_samples),
                'context': context_text,
                'target_word': target_word,
                'predicted_word': predicted_word,
                'confidence': confidence,
                'is_correct': is_correct,
                'in_top_5': target in [p[0] for p in top_k_preds[:5]],
                'top_5_predictions': ', '.join(top_5_words),
            })

        if not all_samples:
            return pd.DataFrame()

        df = pd.DataFrame(all_samples)

        # Stratified sampling if requested
        if stratify_by_confidence and len(df) > num_samples:
            df = self._stratified_sample(df, num_samples)
        else:
            df = df.head(num_samples)

        return df

    def _stratified_sample(self, df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
        # sample evenly from high/med/low confidence, correct/incorrect
        samples_per_stratum = num_samples // 6

        strata = {
            'high_conf_correct': df[(df['confidence'] > 0.5) & df['is_correct']],
            'high_conf_incorrect': df[(df['confidence'] > 0.5) & ~df['is_correct']],
            'med_conf_correct': df[(df['confidence'] > 0.1) & (df['confidence'] <= 0.5) & df['is_correct']],
            'med_conf_incorrect': df[(df['confidence'] > 0.1) & (df['confidence'] <= 0.5) & ~df['is_correct']],
            'low_conf_correct': df[(df['confidence'] <= 0.1) & df['is_correct']],
            'low_conf_incorrect': df[(df['confidence'] <= 0.1) & ~df['is_correct']],
        }

        selected = []
        remaining_quota = num_samples

        for stratum_name, stratum_df in strata.items():
            n_from_stratum = min(samples_per_stratum, len(stratum_df), remaining_quota)
            if n_from_stratum > 0:
                selected.append(stratum_df.sample(n=n_from_stratum, random_state=self.random_seed))
                remaining_quota -= n_from_stratum

        # Fill remaining quota from any stratum
        if remaining_quota > 0:
            used_ids = set()
            for s in selected:
                used_ids.update(s['sample_id'].tolist())

            remaining_df = df[~df['sample_id'].isin(used_ids)]
            if len(remaining_df) > 0:
                n_extra = min(remaining_quota, len(remaining_df))
                selected.append(remaining_df.sample(n=n_extra, random_state=self.random_seed))

        result = pd.concat(selected, ignore_index=True)
        return result.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

    def export_for_rating(self, samples: pd.DataFrame, output_path: str,
                         include_instructions: bool = True):
        # Add rating columns
        df = samples.copy()
        df['grammar_score'] = ''
        df['semantic_score'] = ''
        df['context_appropriateness'] = ''
        df['overall_score'] = ''
        df['notes'] = ''

        # Reorder columns
        column_order = [
            'sample_id', 'context', 'target_word', 'predicted_word',
            'is_correct', 'confidence', 'top_5_predictions',
            'grammar_score', 'semantic_score', 'context_appropriateness',
            'overall_score', 'notes'
        ]
        df = df[[c for c in column_order if c in df.columns]]

        # Save
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"Exported {len(samples)} samples to: {output_path}")

        if include_instructions:
            self._print_rating_instructions()

    def _print_rating_instructions(self):
        print("\n" + "=" * 60)
        print("RATING INSTRUCTIONS")
        print("=" * 60)
        print("""
Please rate each prediction on a scale of 0-3 for each criterion:

GRAMMAR SCORE:
    3 - Perfectly grammatical in context
    2 - Minor grammatical issues but acceptable
    1 - Noticeable grammatical problems
    0 - Ungrammatical or nonsensical

SEMANTIC SCORE:
    3 - Semantically excellent (captures meaning well)
    2 - Semantically reasonable
    1 - Semantically weak but understandable
    0 - Semantically wrong or meaningless

CONTEXT APPROPRIATENESS:
    3 - Perfect fit for the context
    2 - Acceptable alternative to the target
    1 - Somewhat related but not ideal
    0 - Completely inappropriate for context

OVERALL SCORE:
    3 - Excellent (as good as or better than ground truth)
    2 - Acceptable (reasonable alternative)
    1 - Poor (grammatical but semantically off)
    0 - Invalid (ungrammatical or nonsensical)

Use the 'notes' column for any observations or explanations.
""")
        print("=" * 60)

    def analyze_ratings(self, rated_csv_path: str) -> Dict:
        df = pd.read_csv(rated_csv_path)

        # Check required columns
        required_cols = ['overall_score', 'grammar_score', 'semantic_score', 'context_appropriateness']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")

        results = {
            'num_rated': len(df),
            'metrics': {},
        }

        for col in required_cols:
            if col in df.columns:
                # Convert to numeric, ignoring non-numeric values
                scores = pd.to_numeric(df[col], errors='coerce').dropna()

                if len(scores) > 0:
                    results['metrics'][col] = {
                        'mean': float(scores.mean()),
                        'std': float(scores.std()),
                        'median': float(scores.median()),
                        'min': float(scores.min()),
                        'max': float(scores.max()),
                        'distribution': scores.value_counts().sort_index().to_dict(),
                    }

        # Analyze by correctness
        if 'is_correct' in df.columns and 'overall_score' in df.columns:
            overall_scores = pd.to_numeric(df['overall_score'], errors='coerce')

            correct_scores = overall_scores[df['is_correct'] == True].dropna()
            incorrect_scores = overall_scores[df['is_correct'] == False].dropna()

            if len(correct_scores) > 0:
                results['correct_predictions'] = {
                    'count': int(len(correct_scores)),
                    'mean_score': float(correct_scores.mean()),
                }

            if len(incorrect_scores) > 0:
                results['incorrect_predictions'] = {
                    'count': int(len(incorrect_scores)),
                    'mean_score': float(incorrect_scores.mean()),
                }

        # Analyze by confidence level
        if 'confidence' in df.columns and 'overall_score' in df.columns:
            overall_scores = pd.to_numeric(df['overall_score'], errors='coerce')

            high_conf = overall_scores[df['confidence'] > 0.5].dropna()
            low_conf = overall_scores[df['confidence'] <= 0.1].dropna()

            if len(high_conf) > 0:
                results['high_confidence'] = {
                    'count': int(len(high_conf)),
                    'mean_score': float(high_conf.mean()),
                }

            if len(low_conf) > 0:
                results['low_confidence'] = {
                    'count': int(len(low_conf)),
                    'mean_score': float(low_conf.mean()),
                }

        return results

    def print_analysis(self, analysis: Dict):
        print("\n" + "=" * 60)
        print("QUALITATIVE EVALUATION ANALYSIS")
        print("=" * 60)

        print(f"\nTotal samples rated: {analysis['num_rated']}")

        print("\n--- Score Distributions ---")
        for metric, data in analysis.get('metrics', {}).items():
            print(f"\n{metric}:")
            print(f"  Mean: {data['mean']:.2f} ± {data['std']:.2f}")
            print(f"  Median: {data['median']:.2f}")
            print(f"  Range: [{data['min']:.0f}, {data['max']:.0f}]")
            print(f"  Distribution: {data['distribution']}")

        if 'correct_predictions' in analysis:
            print("\n--- By Prediction Correctness ---")
            print(f"Correct predictions: {analysis['correct_predictions']['count']}, "
                  f"mean score: {analysis['correct_predictions']['mean_score']:.2f}")

        if 'incorrect_predictions' in analysis:
            print(f"Incorrect predictions: {analysis['incorrect_predictions']['count']}, "
                  f"mean score: {analysis['incorrect_predictions']['mean_score']:.2f}")

        if 'high_confidence' in analysis:
            print("\n--- By Confidence Level ---")
            print(f"High confidence (>0.5): {analysis['high_confidence']['count']}, "
                  f"mean score: {analysis['high_confidence']['mean_score']:.2f}")

        if 'low_confidence' in analysis:
            print(f"Low confidence (<=0.1): {analysis['low_confidence']['count']}, "
                  f"mean score: {analysis['low_confidence']['mean_score']:.2f}")

    def generate_report_examples(self, samples: pd.DataFrame,
                                num_examples: int = 5) -> List[Dict]:
        # pick diverse examples for report
        examples = []

        # Get some correct predictions
        correct = samples[samples['is_correct'] == True]
        if len(correct) > 0:
            n_correct = min(2, len(correct))
            for _, row in correct.sample(n=n_correct, random_state=self.random_seed).iterrows():
                examples.append(self._row_to_example(row))

        # Get some incorrect but close (in top-5)
        close = samples[(samples['is_correct'] == False) & (samples['in_top_5'] == True)]
        if len(close) > 0:
            n_close = min(2, len(close))
            for _, row in close.sample(n=n_close, random_state=self.random_seed).iterrows():
                examples.append(self._row_to_example(row))

        # Get some completely wrong
        wrong = samples[(samples['is_correct'] == False) & (samples['in_top_5'] == False)]
        if len(wrong) > 0:
            n_wrong = min(1, len(wrong))
            for _, row in wrong.sample(n=n_wrong, random_state=self.random_seed).iterrows():
                examples.append(self._row_to_example(row))

        return examples[:num_examples]

    def _row_to_example(self, row) -> Dict:
        return {
            'context': row['context'],
            'target': row['target_word'],
            'prediction': row['predicted_word'],
            'is_correct': row['is_correct'],
            'confidence': row['confidence'],
            'top_5': row['top_5_predictions'],
        }


def main():
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from models import NgramModel, LSTMModel, GRUModel
    from utils.data_utils import load_processed_data

    parser = argparse.ArgumentParser(description="Qualitative Evaluation Tools")

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Generate samples command
    gen_parser = subparsers.add_parser('generate', help='Generate samples for rating')
    gen_parser.add_argument('--model', required=True, help='Path to model checkpoint')
    gen_parser.add_argument('--data-dir', default='processed_data', help='Data directory')
    gen_parser.add_argument('--output', required=True, help='Output CSV path')
    gen_parser.add_argument('--num-samples', type=int, default=100, help='Number of samples')
    gen_parser.add_argument('--stratify', action='store_true', help='Stratify by confidence')

    # Analyze ratings command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze completed ratings')
    analyze_parser.add_argument('--input', required=True, help='Path to rated CSV')
    analyze_parser.add_argument('--output', help='Output JSON path for analysis')

    args = parser.parse_args()

    if args.command == 'generate':
        # Load data and model
        vocab, _, _, test_sequences = load_processed_data(args.data_dir)

        # Load model
        import torch
        if args.model.endswith('.pkl'):
            model = NgramModel.load(args.model)
        else:
            checkpoint = torch.load(args.model, map_location='cpu')
            model_type = checkpoint.get('model_type', 'lstm')
            if model_type == 'gru':
                model = GRUModel.load(args.model)
            else:
                model = LSTMModel.load(args.model)

        # Generate samples
        evaluator = QualitativeEvaluator(model, vocab)
        samples = evaluator.generate_samples(
            test_sequences,
            num_samples=args.num_samples,
            stratify_by_confidence=args.stratify
        )
        evaluator.export_for_rating(samples, args.output)

    elif args.command == 'analyze':
        # Create dummy evaluator for analysis
        evaluator = QualitativeEvaluator(None, None)
        analysis = evaluator.analyze_ratings(args.input)
        evaluator.print_analysis(analysis)

        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"\nAnalysis saved to: {args.output}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
