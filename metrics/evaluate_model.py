"""
Model Evaluation Script using VQA Metrics

Evaluate model predictions on test dataset using comprehensive metrics from ref.
"""

import torch
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from vqa_metrics import (
    VQAAccuracy, VQASoftAccuracy, TopKAccuracy, F1Score,
    MetricCollection, MetricResult
)


def evaluate_batch_classification(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluate a batch using classification metrics (for tensor input).
    
    Args:
        predictions: Model predictions (torch.Tensor, shape: [batch_size] or [batch_size, num_classes])
        targets: Ground truth targets (torch.Tensor, shape: [batch_size])
    
    Returns:
        Dictionary mapping metric names to scores
    """
    # Create classification-only metrics
    metrics = MetricCollection([
        VQAAccuracy(use_soft_accuracy=True),
        VQASoftAccuracy(),
        TopKAccuracy(k=5),
        TopKAccuracy(k=10),
        F1Score(num_classes=3000, average='macro'),
    ])
    
    # Handle logits
    if predictions.dim() == 2:
        predictions = predictions.argmax(dim=-1)
    
    metrics.update(predictions, targets)
    results = metrics.compute()
    
    return {name: result.value for name, result in results.items()}


def evaluate_dataset(
    predictions_list: List[torch.Tensor],
    targets_list: List[torch.Tensor]
) -> Dict[str, float]:
    """
    Evaluate entire dataset.
    
    Args:
        predictions_list: List of prediction batches
        targets_list: List of target batches
    
    Returns:
        Dictionary mapping metric names to scores
    """
    metrics = MetricCollection([
        VQAAccuracy(use_soft_accuracy=True),
        VQASoftAccuracy(),
        TopKAccuracy(k=5),
        TopKAccuracy(k=10),
        F1Score(num_classes=3000, average='macro'),
    ])
    
    for predictions, targets in zip(predictions_list, targets_list):
        # Handle logits
        if predictions.dim() == 2:
            predictions = predictions.argmax(dim=-1)
        
        metrics.update(predictions, targets)
    
    results = metrics.compute()
    return {name: result.value for name, result in results.items()}


def print_metric_results(results: Dict[str, float], title: str = "Evaluation Results") -> None:
    """
    Pretty print metric results.
    
    Args:
        results: Dictionary of metric name -> score
        title: Title for the results
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    
    for name, score in sorted(results.items()):
        if isinstance(score, float):
            print(f"  {name:30s}: {score:.4f}")
        else:
            print(f"  {name:30s}: {score}")
    
    print("=" * 80 + "\n")


def compare_results(
    results_1: Dict[str, float],
    results_2: Dict[str, float],
    label_1: str = "Model 1",
    label_2: str = "Model 2"
) -> None:
    """
    Compare two sets of metric results.
    
    Args:
        results_1: First set of results
        results_2: Second set of results
        label_1: Label for first set
        label_2: Label for second set
    """
    print("\n" + "=" * 100)
    print(f"Comparison: {label_1} vs {label_2}")
    print("=" * 100)
    print(f"{'Metric':<30} {label_1:>30} {label_2:>30} {'Difference':>10}")
    print("-" * 100)
    
    for name in sorted(results_1.keys()):
        if name in results_2:
            score_1 = results_1[name]
            score_2 = results_2[name]
            diff = score_2 - score_1
            diff_str = f"{diff:+.4f}" if isinstance(diff, float) else "N/A"
            print(f"{name:<30} {score_1:>30.4f} {score_2:>30.4f} {diff_str:>10}")
    
    print("=" * 100 + "\n")


def main():
    """Example usage."""
    print("VQA Metrics Evaluation Module (Classification)")
    print("-" * 80)
    
    # Create dummy predictions and targets
    batch_size = 32
    num_classes = 3000
    
    # Simulate model predictions (logits)
    predictions = torch.randn(batch_size, num_classes)
    
    # Simulate ground truth targets
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Evaluate
    print("Evaluating batch...")
    results = evaluate_batch_classification(predictions, targets)
    
    # Print results
    print_metric_results(results, "Classification Metrics")
    
    # Print key metrics
    print("Key Metrics Summary:")
    print(f"  VQA Accuracy: {results.get('vqa_accuracy', 0):.4f}")
    print(f"  VQA Soft Accuracy: {results.get('vqa_soft_accuracy', 0):.4f}")
    print(f"  Top-5 Accuracy: {results.get('top5_accuracy', 0):.4f}")
    print(f"  F1 Score: {results.get('f1_score', 0):.4f}")


if __name__ == "__main__":
    main()
