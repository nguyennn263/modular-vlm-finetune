"""VQA metrics.

Import submodules directly so ``import metrics.<x>`` never eagerly pulls torch:

    from metrics.vqa_metrics import BLEUScore, CIDErScore      # needs torch
    from metrics.cider import Cider                            # legacy, torch-free
"""


def compute_scores(gts, gen):
    """Legacy helper: run the classic metric implementations over (gts, gen)."""
    from metrics.accuracy import Accuracy
    from metrics.bleu import Bleu
    from metrics.cider import Cider
    from metrics.f1 import F1
    from metrics.meteor import Meteor
    from metrics.precision import Precision
    from metrics.recall import Recall
    from metrics.rouge import Rouge
    from metrics.wup import Wup

    metrics = (Bleu(), Meteor(), Rouge(), Cider(), Wup(), Accuracy(), Precision(), Recall(), F1())
    all_score, all_scores = {}, {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores
    return all_score, all_scores
