"""
VQA Metrics - Wrapper around metrics/ module
Sử dụng ScoreCalculator từ metrics/ module cho evaluation
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np

# Add metrics/ to path for importing
METRICS_PATH = Path(__file__).parent.parent.parent / "metrics"
if str(METRICS_PATH) not in sys.path:
    sys.path.insert(0, str(METRICS_PATH))
    # Add subdirectories for proper imports
    for subdir in METRICS_PATH.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.'):
            sys.path.insert(0, str(subdir))


def contains_match(prediction: str, references: Union[str, List[str]]) -> float:
    """
    Simple check if any reference is contained in prediction
    Lightweight utility function
    """
    if isinstance(references, str):
        references = [references]
    
    pred_lower = prediction.lower().strip()
    for ref in references:
        ref_lower = ref.lower().strip()
        if ref_lower in pred_lower or pred_lower in ref_lower:
            return 1.0
    return 0.0


class VQAMetrics:
    """
    Wrapper around metrics/ module ScoreCalculator
    """
    
    def __init__(self, use_full_metrics: bool = False):
        """
        Args:
            use_full_metrics: Sử dụng full metrics từ metrics/ module
        """
        self.use_full_metrics = use_full_metrics
        self._score_calculator = None
        
        if use_full_metrics:
            self._init_calculator()
    
    def _init_calculator(self):
        """Initialize ScoreCalculator từ metrics/ module"""
        try:
            from compute_score import ScoreCalculator
            self._score_calculator = ScoreCalculator()
        except ImportError as e:
            print(f"Warning: Could not import metrics module: {e}")
            self.use_full_metrics = False
    
    def compute(
        self,
        predictions: List[str],
        references: List[List[str]],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute metrics sử dụng ScoreCalculator
        """
        if not self.use_full_metrics or not self._score_calculator:
            return {}
        
        if metrics is None:
            metrics = ["accuracy", "bleu", "f1_token", "rouge"]
        
        results = {}
        
        metric_methods = {
            "accuracy": "accuracy_score",
            "bleu": "bleu_score",
            "cider": "cider_score",
            "f1_token": "f1_token",
            "meteor": "meteor_score",
            "precision": "precision_score",
            "recall": "recall_score",
            "rouge": "rouge_score",
            "wup": "wup",
        }
        
        for metric_name in metrics:
            if metric_name not in metric_methods:
                continue
                
            method_name = metric_methods[metric_name]
            try:
                method = getattr(self._score_calculator, method_name)
                scores = [
                    method(refs, pred) 
                    for pred, refs in zip(predictions, references)
                ]
                results[metric_name] = float(np.mean(scores)) if scores else 0.0
            except Exception as e:
                print(f"Warning: Failed to compute {metric_name}: {e}")
                results[metric_name] = 0.0
        
        return results


def evaluate_model_outputs(
    predictions: List[str],
    references: List[List[str]],
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate model outputs với full metrics từ metrics/ module
    """
    calculator = VQAMetrics(use_full_metrics=True)
    return calculator.compute(predictions, references, metrics)
