"""
VQA Metrics cho đánh giá Vision-Language Model
"""
import re
from typing import List, Dict, Optional
import torch


def normalize_answer(text: str) -> str:
    """Chuẩn hóa câu trả lời để so sánh"""
    text = text.lower().strip()
    text = re.sub(r'[.,!?;:]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def exact_match(prediction: str, reference: str) -> float:
    """Exact match score"""
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def contains_match(prediction: str, reference: str) -> float:
    """Check if reference is contained in prediction"""
    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)
    return 1.0 if ref_norm in pred_norm else 0.0


def compute_vqa_accuracy(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute VQA accuracy metrics
    Returns: dict với exact_match và contains_match scores
    """
    exact_scores = []
    contains_scores = []
    
    for pred, ref in zip(predictions, references):
        exact_scores.append(exact_match(pred, ref))
        contains_scores.append(contains_match(pred, ref))
    
    return {
        "exact_match": sum(exact_scores) / len(exact_scores),
        "contains_match": sum(contains_scores) / len(contains_scores),
    }


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU score (requires sacrebleu)"""
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        return bleu.score / 100.0
    except ImportError:
        return 0.0


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores (requires rouge_score)"""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)
            for key in scores:
                scores[key].append(result[key].fmeasure)
        
        return {k: sum(v) / len(v) for k, v in scores.items()}
    except ImportError:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}


class VQAMetrics:
    """Class để tính toán và theo dõi VQA metrics"""
    
    def __init__(self):
        self.predictions = []
        self.references = []
    
    def add_batch(self, predictions: List[str], references: List[str]):
        """Thêm batch predictions và references"""
        self.predictions.extend(predictions)
        self.references.extend(references)
    
    def compute(self) -> Dict[str, float]:
        """Compute tất cả metrics"""
        results = compute_vqa_accuracy(self.predictions, self.references)
        results.update(compute_rouge(self.predictions, self.references))
        results["bleu"] = compute_bleu(self.predictions, self.references)
        return results
    
    def reset(self):
        """Reset metrics"""
        self.predictions = []
        self.references = []
