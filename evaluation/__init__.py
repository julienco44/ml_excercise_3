"""
Evaluation Module for Next Word Prediction

Contains metrics, semantic similarity evaluation, statistical tests,
and qualitative evaluation tools.
"""

from .metrics import (
    calculate_accuracy,
    calculate_topk_accuracy,
    calculate_perplexity,
    calculate_mrr,
    evaluate_model,
    print_evaluation_results,
    generate_sample_predictions,
)
from .semantic_similarity import SemanticEvaluator
from .statistical_tests import paired_ttest, bootstrap_ci, cohens_d, compare_models
from .qualitative_eval import QualitativeEvaluator

__all__ = [
    'calculate_accuracy',
    'calculate_topk_accuracy',
    'calculate_perplexity',
    'calculate_mrr',
    'evaluate_model',
    'print_evaluation_results',
    'generate_sample_predictions',
    'SemanticEvaluator',
    'paired_ttest',
    'bootstrap_ci',
    'cohens_d',
    'compare_models',
    'QualitativeEvaluator',
]
