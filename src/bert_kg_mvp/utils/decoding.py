from typing import Tuple
import torch


def decode_relation_predictions(
    rel_logits: torch.Tensor,
    no_relation_idx: int,
    confidence_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Decodes relation predictions from relation logits using DETR-style aggregate
    foreground confidence thresholding.

    Instead of hard argmax over all classes (where no_relation wins by plurality against
    distributed probability mass across multiple foreground classes), a query is classified
    as foreground if the probability of no_relation does not exceed confidence_threshold
    (e.g., P(no_relation) <= 0.5, meaning aggregate foreground probability >= 0.5).

    Args:
        rel_logits: Tensor of shape [..., num_classes] (where num_classes = num_relations + 1).
        no_relation_idx: Index of the no_relation class (typically num_relations).
        confidence_threshold: Threshold above which a query is classified as background (no_relation).
                              Defaults to 0.5.

    Returns:
        Tensor of shape [...] with predicted class indices.
    """
    rel_probs = torch.softmax(rel_logits, dim=-1)
    no_rel_prob = rel_probs[..., no_relation_idx]

    # Mask out no_relation to identify the highest-probability foreground class
    fg_probs = rel_probs.clone()
    fg_probs[..., no_relation_idx] = -1.0
    best_fg = fg_probs.argmax(dim=-1)

    # Classify as background if P(no_relation) exceeds threshold
    is_background = no_rel_prob > confidence_threshold

    rel_preds = best_fg.clone()
    rel_preds[is_background] = no_relation_idx
    return rel_preds


def compute_relation_diagnostics(
    rel_logits: torch.Tensor,
    no_relation_idx: int,
) -> Tuple[float, float, float]:
    """
    Computes diagnostic confidence statistics over query relation predictions.

    Args:
        rel_logits: Tensor of shape [..., num_classes].
        no_relation_idx: Index of the no_relation class.

    Returns:
        Tuple of (mean_fg_prob, max_fg_prob, mean_no_rel_prob)
    """
    rel_probs = torch.softmax(rel_logits, dim=-1)
    no_rel_prob = rel_probs[..., no_relation_idx]
    fg_sum = 1.0 - no_rel_prob

    mean_fg = fg_sum.mean().item()
    max_fg = fg_sum.max().item()
    mean_no_rel = no_rel_prob.mean().item()

    return mean_fg, max_fg, mean_no_rel
