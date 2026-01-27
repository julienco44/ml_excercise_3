# Statistical tests for comparing models

from typing import List, Tuple
import numpy as np
from scipy import stats


def paired_ttest(results1: List[float], results2: List[float]) -> Tuple[float, float]:
    """Paired t-test for cross-validation results."""
    if len(results1) != len(results2):
        raise ValueError("Both result lists must have the same length")
    t_stat, p_value = stats.ttest_rel(results1, results2)
    return float(t_stat), float(p_value)


def bootstrap_ci(data: List[float], num_bootstrap: int = 10000,
                confidence: float = 0.95) -> Tuple[float, float]:
    """Bootstrap confidence interval."""
    data = np.array(data)
    n = len(data)

    bootstrap_means = []
    for _ in range(num_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return float(lower), float(upper)


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Cohen's d effect size."""
    group1 = np.array(group1)
    group2 = np.array(group2)

    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    n1, n2 = len(group1), len(group2)

    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((mean1 - mean2) / pooled_std)


def interpret_cohens_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def compare_models(model1_results: List[float], model2_results: List[float],
                  model1_name: str = "Model 1", model2_name: str = "Model 2",
                  metric_name: str = "metric") -> dict:
    """Full comparison with t-test, effect size, and CIs."""
    mean1, std1 = np.mean(model1_results), np.std(model1_results, ddof=1)
    mean2, std2 = np.mean(model2_results), np.std(model2_results, ddof=1)

    # Paired t-test
    t_stat, p_value = paired_ttest(model1_results, model2_results)

    # Effect size
    d = cohens_d(model1_results, model2_results)

    # Bootstrap CIs
    ci1 = bootstrap_ci(model1_results)
    ci2 = bootstrap_ci(model2_results)

    # Determine significance
    is_significant = p_value < 0.05
    better_model = model1_name if mean1 > mean2 else model2_name

    return {
        'metric': metric_name,
        'model1': {
            'name': model1_name,
            'mean': mean1,
            'std': std1,
            'ci_95': ci1,
        },
        'model2': {
            'name': model2_name,
            'mean': mean2,
            'std': std2,
            'ci_95': ci2,
        },
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': d,
        'effect_size_interpretation': interpret_cohens_d(d),
        'is_significant': is_significant,
        'better_model': better_model if is_significant else "No significant difference",
    }


def print_comparison(comparison: dict):
    print(f"\n{'=' * 60}")
    print(f"Statistical Comparison: {comparison['metric']}")
    print('=' * 60)

    m1 = comparison['model1']
    m2 = comparison['model2']

    print(f"\n{m1['name']}:")
    print(f"  Mean: {m1['mean']:.4f} +/- {m1['std']:.4f}")
    print(f"  95% CI: [{m1['ci_95'][0]:.4f}, {m1['ci_95'][1]:.4f}]")

    print(f"\n{m2['name']}:")
    print(f"  Mean: {m2['mean']:.4f} +/- {m2['std']:.4f}")
    print(f"  95% CI: [{m2['ci_95'][0]:.4f}, {m2['ci_95'][1]:.4f}]")

    print(f"\nStatistical Tests:")
    print(f"  t-statistic: {comparison['t_statistic']:.4f}")
    print(f"  p-value: {comparison['p_value']:.4f}")
    print(f"  Cohen's d: {comparison['cohens_d']:.4f} ({comparison['effect_size_interpretation']})")

    if comparison['is_significant']:
        print(f"\n  Result: {comparison['better_model']} is significantly better (p < 0.05)")
    else:
        print(f"\n  Result: No statistically significant difference (p >= 0.05)")
