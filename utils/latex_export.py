# Generate LaTeX tables from results

import os
from typing import Dict, List, Optional, Any
import pandas as pd


def format_metric(value: float, metric_type: str = 'accuracy',
                 include_percent: bool = False) -> str:
    if metric_type == 'perplexity':
        return f"{value:.2f}"
    elif metric_type == 'time':
        return f"{value:.1f}"
    elif metric_type in ['accuracy', 'mrr']:
        if include_percent:
            return f"{value * 100:.2f}\\%"
        else:
            return f"{value:.4f}"
    elif metric_type == 'parameters':
        if value >= 1e6:
            return f"{value / 1e6:.1f}M"
        elif value >= 1e3:
            return f"{value / 1e3:.1f}K"
        else:
            return f"{value:.0f}"
    else:
        return f"{value:.4f}"


def format_mean_std(mean: float, std: float, metric_type: str = 'accuracy') -> str:
    if metric_type == 'perplexity':
        return f"{mean:.2f} $\\pm$ {std:.2f}"
    elif metric_type in ['accuracy', 'mrr']:
        return f"{mean:.4f} $\\pm$ {std:.4f}"
    else:
        return f"{mean:.4f} $\\pm$ {std:.4f}"


def generate_model_comparison_table(results: List[Dict],
                                   caption: str = "Model Performance Comparison",
                                   label: str = "tab:model_comparison",
                                   output_path: Optional[str] = None,
                                   highlight_best: bool = True) -> str:
    rows = []
    for r in results:
        row = {
            'Model': r.get('model_name', 'Unknown'),
            'Perplexity': r.get('perplexity', float('inf')),
            'Top-1': r.get('accuracy', r.get('top_1_accuracy', 0)),
            'Top-5': r.get('top_5_accuracy', 0),
            'MRR': r.get('mrr', 0),
        }
        if 'parameters' in r:
            row['Params'] = r['parameters']
        rows.append(row)

    df = pd.DataFrame(rows)

    # Find best values
    best_values = {}
    if highlight_best:
        best_values['Perplexity'] = df['Perplexity'].min()
        best_values['Top-1'] = df['Top-1'].max()
        best_values['Top-5'] = df['Top-5'].max()
        best_values['MRR'] = df['MRR'].max()

    # Build LaTeX
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")

    # Determine columns
    has_params = 'Params' in df.columns
    if has_params:
        col_format = "lccccc"
        header = "Model & Perplexity & Top-1 Acc & Top-5 Acc & MRR & Params \\\\"
    else:
        col_format = "lcccc"
        header = "Model & Perplexity & Top-1 Acc & Top-5 Acc & MRR \\\\"

    lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")

    # Data rows
    for _, row in df.iterrows():
        cells = [row['Model']]

        # Perplexity
        val = format_metric(row['Perplexity'], 'perplexity')
        if highlight_best and row['Perplexity'] == best_values['Perplexity']:
            val = f"\\textbf{{{val}}}"
        cells.append(val)

        # Top-1
        val = f"{row['Top-1']*100:.2f}\\%"
        if highlight_best and row['Top-1'] == best_values['Top-1']:
            val = f"\\textbf{{{val}}}"
        cells.append(val)

        # Top-5
        val = f"{row['Top-5']*100:.2f}\\%"
        if highlight_best and row['Top-5'] == best_values['Top-5']:
            val = f"\\textbf{{{val}}}"
        cells.append(val)

        # MRR
        val = f"{row['MRR']:.3f}"
        if highlight_best and row['MRR'] == best_values['MRR']:
            val = f"\\textbf{{{val}}}"
        cells.append(val)

        # Params
        if has_params:
            cells.append(format_metric(row['Params'], 'parameters'))

        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to: {output_path}")

    return latex


def generate_cv_results_table(cv_results: List[Dict],
                             caption: str = "Cross-Validation Results (Mean $\\pm$ Std Dev)",
                             label: str = "tab:cv_results",
                             output_path: Optional[str] = None) -> str:
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Model & Perplexity & Top-1 Accuracy & Top-5 Accuracy & MRR \\\\")
    lines.append("\\midrule")

    for r in cv_results:
        cells = [r.get('model_name', 'Unknown')]

        # Perplexity
        mean = r.get('perplexity_mean', 0)
        std = r.get('perplexity_std', 0)
        cells.append(f"{mean:.2f} $\\pm$ {std:.2f}")

        # Top-1 Accuracy
        mean = r.get('accuracy_mean', 0) * 100
        std = r.get('accuracy_std', 0) * 100
        cells.append(f"{mean:.2f}\\% $\\pm$ {std:.2f}\\%")

        # Top-5 Accuracy
        mean = r.get('top_5_accuracy_mean', 0) * 100
        std = r.get('top_5_accuracy_std', 0) * 100
        cells.append(f"{mean:.2f}\\% $\\pm$ {std:.2f}\\%")

        # MRR
        mean = r.get('mrr_mean', 0)
        std = r.get('mrr_std', 0)
        cells.append(f"{mean:.4f} $\\pm$ {std:.4f}")

        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to: {output_path}")

    return latex


def generate_ngram_comparison_table(results: List[Dict],
                                   caption: str = "N-gram Model Comparison",
                                   label: str = "tab:ngram_comparison",
                                   output_path: Optional[str] = None) -> str:
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("Model & Smoothing & Perplexity & Top-1 Acc & Top-5 Acc & MRR \\\\")
    lines.append("\\midrule")

    # Find best values
    perplexities = [r.get('perplexity', float('inf')) for r in results]
    accuracies = [r.get('accuracy', 0) for r in results]
    best_ppl = min(perplexities)
    best_acc = max(accuracies)

    for r in results:
        n = r.get('n', 3)
        smoothing = r.get('smoothing', 'unknown').replace('_', '-').title()

        cells = [f"{n}-gram", smoothing]

        # Perplexity
        ppl = r.get('perplexity', 0)
        val = f"{ppl:.2f}"
        if ppl == best_ppl:
            val = f"\\textbf{{{val}}}"
        cells.append(val)

        # Accuracy
        acc = r.get('accuracy', 0)
        val = f"{acc*100:.2f}\\%"
        if acc == best_acc:
            val = f"\\textbf{{{val}}}"
        cells.append(val)

        # Top-5
        cells.append(f"{r.get('top_5_accuracy', 0)*100:.2f}\\%")

        # MRR
        cells.append(f"{r.get('mrr', 0):.3f}")

        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to: {output_path}")

    return latex


def generate_example_predictions_table(examples: List[Dict],
                                       caption: str = "Example Predictions",
                                       label: str = "tab:examples",
                                       output_path: Optional[str] = None,
                                       max_context_words: int = 8) -> str:
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\begin{tabular}{p{6cm}ccp{3cm}}")
    lines.append("\\toprule")
    lines.append("Context & Target & Correct? & Top Predictions \\\\")
    lines.append("\\midrule")

    for ex in examples:
        # Truncate context
        context = ex.get('context', '')
        words = context.split()
        if len(words) > max_context_words:
            context = '...' + ' '.join(words[-max_context_words:])

        # Escape special characters
        context = context.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')

        target = ex.get('target', '').replace('_', '\\_')
        is_correct = ex.get('is_correct', False)

        # Get predictions
        if 'predictions' in ex and isinstance(ex['predictions'], list):
            preds = [p[0] if isinstance(p, tuple) else p for p in ex['predictions'][:3]]
        elif 'top_5' in ex:
            preds = ex['top_5'].split(', ')[:3]
        else:
            preds = [ex.get('prediction', '')]

        pred_str = ', '.join(preds).replace('_', '\\_')

        correct_mark = "$\\checkmark$" if is_correct else "$\\times$"

        cells = [f"``{context}''", target, correct_mark, pred_str]
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to: {output_path}")

    return latex


def generate_hyperparameter_table(results: List[Dict],
                                 hyperparameter: str,
                                 caption: str = "Hyperparameter Sensitivity",
                                 label: str = "tab:hyperparam",
                                 output_path: Optional[str] = None) -> str:
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\begin{tabular}{ccccc}")
    lines.append("\\toprule")
    lines.append(f"{hyperparameter.title()} & Perplexity & Top-1 Acc & Top-5 Acc & MRR \\\\")
    lines.append("\\midrule")

    for r in results:
        hp_val = r.get(hyperparameter, 'N/A')
        cells = [str(hp_val)]
        cells.append(f"{r.get('perplexity', 0):.2f}")
        cells.append(f"{r.get('accuracy', 0)*100:.2f}\\%")
        cells.append(f"{r.get('top_5_accuracy', 0)*100:.2f}\\%")
        cells.append(f"{r.get('mrr', 0):.3f}")

        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to: {output_path}")

    return latex


def results_to_latex_row(results: Dict, model_name: str = None) -> str:
    name = model_name or results.get('model_name', 'Model')
    ppl = results.get('perplexity', 0)
    acc = results.get('accuracy', 0) * 100
    top5 = results.get('top_5_accuracy', 0) * 100
    top10 = results.get('top_10_accuracy', 0) * 100
    mrr = results.get('mrr', 0)
    params = results.get('parameters', '-')

    if isinstance(params, (int, float)):
        params = format_metric(params, 'parameters')

    return f"{name} & {ppl:.2f} & {acc:.2f}\\% & {top5:.2f}\\% & {top10:.2f}\\% & {mrr:.3f} & {params} \\\\"
