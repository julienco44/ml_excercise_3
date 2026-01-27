# Plotting functions for training and evaluation

import os
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_training_history(history: Dict, save_path: Optional[str] = None,
                         title: str = "Training History"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1 = axes[0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Perplexity plot
    ax2 = axes[1]
    if 'val_perplexity' in history and history['val_perplexity']:
        ax2.plot(epochs, history['val_perplexity'], 'g-', label='Val Perplexity')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('Validation Perplexity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No validation perplexity data',
                ha='center', va='center', transform=ax2.transAxes)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.close()


def plot_model_comparison(results: List[Dict], metric: str = 'accuracy',
                         save_path: Optional[str] = None,
                         title: str = "Model Comparison"):
    fig, ax = plt.subplots(figsize=(10, 6))

    models = [r.get('model_name', r.get('config_name', f"Model {i}"))
              for i, r in enumerate(results)]
    values = [r[metric] for r in results]

    # Sort by value
    sorted_pairs = sorted(zip(models, values), key=lambda x: x[1], reverse=True)
    models, values = zip(*sorted_pairs)

    bars = ax.barh(models, values, color='steelblue')

    # Add value labels
    for bar, val in zip(bars, values):
        if metric in ['accuracy', 'top_3_accuracy', 'top_5_accuracy', 'top_10_accuracy', 'mrr']:
            label = f'{val:.2%}'
        else:
            label = f'{val:.2f}'
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               label, va='center', fontsize=9)

    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.close()


def plot_perplexity_comparison(results: List[Dict], save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(10, 6))

    models = [r.get('model_name', r.get('config_name', f"Model {i}"))
              for i, r in enumerate(results)]
    perplexities = [r['perplexity'] for r in results]

    # Sort by perplexity (ascending)
    sorted_pairs = sorted(zip(models, perplexities), key=lambda x: x[1])
    models, perplexities = zip(*sorted_pairs)

    bars = ax.barh(models, perplexities, color='coral')

    # Add value labels
    for bar, val in zip(bars, perplexities):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}', va='center', fontsize=9)

    ax.set_xlabel('Perplexity (lower is better)')
    ax.set_title('Model Perplexity Comparison')
    ax.set_xscale('log')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.close()


def plot_all_metrics(results: List[Dict], save_path: Optional[str] = None):
    metrics = ['accuracy', 'top_5_accuracy', 'mrr', 'perplexity']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        models = [r.get('model_name', r.get('config_name', f"Model {i}"))
                  for i, r in enumerate(results)]
        values = [r.get(metric, 0) for r in results]

        # Sort appropriately
        if metric == 'perplexity':
            sorted_pairs = sorted(zip(models, values), key=lambda x: x[1])
        else:
            sorted_pairs = sorted(zip(models, values), key=lambda x: x[1], reverse=True)

        models_sorted, values_sorted = zip(*sorted_pairs)

        colors = 'coral' if metric == 'perplexity' else 'steelblue'
        bars = ax.barh(models_sorted, values_sorted, color=colors)

        # Add value labels
        for bar, val in zip(bars, values_sorted):
            if metric == 'perplexity':
                label = f'{val:.1f}'
            else:
                label = f'{val:.2%}'
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                   label, va='center', fontsize=8)

        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(True, axis='x', alpha=0.3)

    plt.suptitle('Model Comparison - All Metrics', fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.close()


def plot_confusion_by_position(results: Dict, save_path: Optional[str] = None):
    if 'position_accuracy' not in results:
        print("No position accuracy data available")
        return

    positions = list(range(len(results['position_accuracy'])))
    accuracies = results['position_accuracy']

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(positions, accuracies, 'b-o', markersize=4)
    ax.set_xlabel('Position in Sequence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Prediction Accuracy by Position')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.close()
