# Import from ref's comprehensive VQA metrics module
from .vqa_metrics import (
    # Data structures
    MetricResult,
    # Base class
    BaseMetric,
    # Accuracy metrics
    VQAAccuracy,
    VQASoftAccuracy,
    TopKAccuracy,
    AnswerTypeAccuracy,
    ExactMatchAccuracy,
    # Classification metrics
    F1Score,
    # Generation metrics
    BLEUScore,
    METEORScore,
    ROUGEScore,
    CIDErScore,
    # Semantic metrics
    WUPS,
    PrecisionRecallF1,
    # Metric management
    MetricCollection,
    # Factory functions
    create_vqa_metrics,
    create_comprehensive_vqa_metrics,
)

# Legacy imports (for backward compatibility)
from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider
from .accuracy import Accuracy
from .f1 import F1
from .precision import Precision
from .recall import Recall
from .wup import Wup


def compute_scores(gts, gen):
    """
    Legacy compute_scores function using old metric implementations.
    For new code, use create_vqa_metrics() or create_comprehensive_vqa_metrics() instead.
    """
    metrics = (Bleu(), Meteor(), Rouge(), Cider(), Wup(), Accuracy(), Precision(), Recall(), F1())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores


def compute_vqa_metrics(predictions, targets, metric_type='standard', num_classes=3000, answer_types=None, id2answer=None):
    """
    Compute VQA metrics using the comprehensive ref metrics module.
    
    Args:
        predictions: Model predictions (torch.Tensor or list)
        targets: Ground truth targets (torch.Tensor or list)
        metric_type: 'standard' or 'comprehensive'
        num_classes: Number of answer classes (for F1Score)
        answer_types: List of answer types (for AnswerTypeAccuracy)
        id2answer: Mapping from answer ID to string (optional)
    
    Returns:
        Dictionary of metric results
    """
    if metric_type == 'comprehensive':
        metrics = create_comprehensive_vqa_metrics(id2answer=id2answer)
    else:
        metrics = create_vqa_metrics(
            num_classes=num_classes,
            answer_types=answer_types,
            id2answer=id2answer
        )
    
    # Update metrics with data
    metrics.update(predictions, targets)
    
    # Return computed results
    results = metrics.compute()
    return {name: result.value for name, result in results.items()}


__all__ = [
    # New ref metrics
    'MetricResult',
    'BaseMetric',
    'VQAAccuracy',
    'VQASoftAccuracy',
    'TopKAccuracy',
    'AnswerTypeAccuracy',
    'ExactMatchAccuracy',
    'F1Score',
    'BLEUScore',
    'METEORScore',
    'ROUGEScore',
    'CIDErScore',
    'WUPS',
    'PrecisionRecallF1',
    'MetricCollection',
    'create_vqa_metrics',
    'create_comprehensive_vqa_metrics',
    # Legacy metrics (deprecated)
    'Bleu',
    'Meteor',
    'Rouge',
    'Cider',
    'Accuracy',
    'F1',
    'Precision',
    'Recall',
    'Wup',
    # Functions
    'compute_scores',
    'compute_vqa_metrics',
]

