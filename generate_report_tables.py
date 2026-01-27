#!/usr/bin/env python3
"""
Generate Report Tables

Aggregates all experimental results and generates LaTeX tables for the report.
"""

import os
import sys
import json
import glob
import argparse
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.latex_export import (
    generate_model_comparison_table,
    generate_cv_results_table,
    generate_ngram_comparison_table,
    generate_example_predictions_table,
    generate_hyperparameter_table,
    results_to_latex_row,
)


def load_results(results_dir: str = 'experiments/results') -> Dict:
    """
    Load all experimental results from the results directory.

    Args:
        results_dir: Path to results directory

    Returns:
        Dictionary with all results organized by type
    """
    all_results = {
        'ngram': [],
        'neural': [],
        'cv': [],
        'samples': [],
    }

    # Load JSON result files
    json_files = glob.glob(os.path.join(results_dir, '*.json'))

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue

        filename = os.path.basename(json_file)

        # Categorize results
        if '_cv_results' in filename:
            all_results['cv'].append(data)
        elif 'ngram' in filename.lower() or 'gram' in filename.lower():
            if 'validation_results' in data:
                result = data['validation_results'].copy()
                result['config'] = data.get('config', {})
                all_results['ngram'].append(result)
            elif 'results' in data:
                for r in data['results']:
                    all_results['ngram'].append(r)
        elif 'evaluation_results' in filename:
            if 'results' in data:
                for r in data['results']:
                    all_results['neural'].append(r)
            if 'sample_predictions' in data:
                all_results['samples'] = data['sample_predictions']
        else:
            # Try to identify result type from content
            if 'validation_results' in data:
                result = data['validation_results'].copy()
                result['config'] = data.get('config', {})
                if data.get('config', {}).get('model_type', '').lower() == 'ngram':
                    all_results['ngram'].append(result)
                else:
                    all_results['neural'].append(result)

    return all_results


def load_ngram_results(ngram_results_dir: str = 'ngram_results') -> List[Dict]:
    """Load N-gram specific results."""
    results = []

    # Check for comparison results
    comparison_file = os.path.join(ngram_results_dir, 'all_comparisons.json')
    if os.path.exists(comparison_file):
        with open(comparison_file, 'r') as f:
            data = json.load(f)
            if 'results' in data:
                for r in data['results']:
                    results.append(r)

    # Check individual result files
    json_files = glob.glob(os.path.join(ngram_results_dir, '*_results.json'))
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if 'validation_results' in data:
                    result = data['validation_results'].copy()
                    config = data.get('config', {})
                    result['n'] = config.get('n', 3)
                    result['smoothing'] = config.get('smoothing', 'unknown')
                    results.append(result)
        except:
            continue

    return results


def generate_all_tables(results: Dict, output_dir: str = 'experiments/results/latex_tables'):
    """
    Generate all LaTeX tables for the report.

    Args:
        results: Dictionary of all results
        output_dir: Directory to save LaTeX files
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("GENERATING LATEX TABLES")
    print("=" * 60)

    # 1. N-gram comparison table
    if results['ngram']:
        print("\n1. Generating N-gram comparison table...")
        generate_ngram_comparison_table(
            results['ngram'],
            caption="N-gram Model Performance Comparison",
            label="tab:ngram_comparison",
            output_path=os.path.join(output_dir, 'ngram_comparison.tex')
        )
    else:
        print("\n1. No N-gram results found, skipping...")

    # 2. Main model comparison table
    all_models = results['ngram'] + results['neural']
    if all_models:
        print("\n2. Generating main model comparison table...")
        generate_model_comparison_table(
            all_models,
            caption="Complete Model Performance Comparison on Test Set",
            label="tab:model_comparison",
            output_path=os.path.join(output_dir, 'model_comparison.tex')
        )
    else:
        print("\n2. No model results found, skipping...")

    # 3. Cross-validation results
    if results['cv']:
        print("\n3. Generating cross-validation results table...")
        generate_cv_results_table(
            results['cv'],
            caption="Cross-Validation Results (Mean $\\pm$ Std Dev)",
            label="tab:cv_results",
            output_path=os.path.join(output_dir, 'cv_results.tex')
        )
    else:
        print("\n3. No cross-validation results found, skipping...")

    # 4. Example predictions table
    if results['samples']:
        print("\n4. Generating example predictions table...")
        # Flatten samples from all models
        all_samples = []
        for model_name, samples in results['samples'].items():
            if isinstance(samples, list):
                for s in samples[:3]:  # Take up to 3 from each model
                    s['model'] = model_name
                    all_samples.append(s)

        if all_samples:
            generate_example_predictions_table(
                all_samples[:10],
                caption="Example Predictions from Best Models",
                label="tab:examples",
                output_path=os.path.join(output_dir, 'example_predictions.tex')
            )
    else:
        print("\n4. No sample predictions found, skipping...")

    print("\n" + "=" * 60)
    print(f"LaTeX tables saved to: {output_dir}")
    print("=" * 60)


def generate_summary_table(results: Dict, output_path: Optional[str] = None) -> str:
    """
    Generate a summary table combining key results.

    Args:
        results: Dictionary of all results
        output_path: Optional path to save

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Summary of Model Performance}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Model & Perplexity & Top-1 Acc & Top-5 Acc & Top-10 Acc & MRR & Parameters \\\\")
    lines.append("\\midrule")

    # Add rows for each model
    all_models = results.get('ngram', []) + results.get('neural', [])

    for r in all_models:
        lines.append(results_to_latex_row(r))

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\label{tab:summary}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"Summary table saved to: {output_path}")

    return latex


def print_results_summary(results: Dict):
    """Print a human-readable summary of results."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nN-gram models: {len(results['ngram'])}")
    for r in results['ngram']:
        name = r.get('model_name', f"{r.get('n', '?')}-gram ({r.get('smoothing', '?')})")
        print(f"  - {name}: Perplexity={r.get('perplexity', 'N/A'):.2f}, "
              f"Accuracy={r.get('accuracy', 0)*100:.2f}%")

    print(f"\nNeural models: {len(results['neural'])}")
    for r in results['neural']:
        name = r.get('model_name', 'Unknown')
        print(f"  - {name}: Perplexity={r.get('perplexity', 'N/A'):.2f}, "
              f"Accuracy={r.get('accuracy', 0)*100:.2f}%")

    print(f"\nCross-validation results: {len(results['cv'])}")
    for r in results['cv']:
        name = r.get('model_name', 'Unknown')
        print(f"  - {name}: Perplexity={r.get('perplexity_mean', 'N/A'):.2f} ± "
              f"{r.get('perplexity_std', 0):.2f}")

    # Find best models
    all_models = results['ngram'] + results['neural']
    if all_models:
        best_ppl = min(all_models, key=lambda x: x.get('perplexity', float('inf')))
        best_acc = max(all_models, key=lambda x: x.get('accuracy', 0))

        print("\n--- Best Models ---")
        print(f"Best Perplexity: {best_ppl.get('model_name', 'Unknown')} "
              f"({best_ppl.get('perplexity', 'N/A'):.2f})")
        print(f"Best Accuracy: {best_acc.get('model_name', 'Unknown')} "
              f"({best_acc.get('accuracy', 0)*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from results")

    parser.add_argument('--results-dir', type=str, default='experiments/results',
                       help='Directory containing result JSON files')
    parser.add_argument('--ngram-dir', type=str, default='ngram_results',
                       help='Directory containing N-gram results')
    parser.add_argument('--output-dir', type=str, default='experiments/results/latex_tables',
                       help='Output directory for LaTeX files')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only print summary, do not generate tables')

    args = parser.parse_args()

    # Load all results
    print("Loading results...")
    results = load_results(args.results_dir)

    # Also load N-gram specific results
    ngram_specific = load_ngram_results(args.ngram_dir)

    # Merge N-gram results, avoiding duplicates
    existing_names = {r.get('model_name', '') for r in results['ngram']}
    for r in ngram_specific:
        name = r.get('model_name', f"{r.get('n', '?')}-gram")
        if name not in existing_names:
            results['ngram'].append(r)

    # Print summary
    print_results_summary(results)

    if not args.summary_only:
        # Generate tables
        generate_all_tables(results, args.output_dir)

        # Generate summary table
        summary_path = os.path.join(args.output_dir, 'summary.tex')
        generate_summary_table(results, summary_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
