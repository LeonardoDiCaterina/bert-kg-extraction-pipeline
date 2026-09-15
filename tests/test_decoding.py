import torch
import pytest
from torch.utils.data import TensorDataset

from bert_kg_mvp.utils.decoding import (
    decode_relation_predictions,
    compute_relation_diagnostics,
)
from bert_kg_mvp.pipelines.benchmark.nodes import evaluate_model_on_test


def test_decode_relation_predictions_argmax_trap():
    """
    Tests the critical failure mode:
    P(no_relation) = 0.35 is the single highest class, but aggregate foreground is 0.65.
    Hard argmax would choose no_relation (index 13).
    With confidence_threshold=0.5, it correctly predicts rel_0 (highest foreground).
    """
    num_relations = 13
    no_relation_idx = num_relations
    
    # Batch size 1, 1 query slot, 14 classes
    # Logits calibrated to: rel_0 ~ 0.30, rel_1 ~ 0.20, rel_2..12 ~ 0.15 / 11, no_rel ~ 0.35
    probs = torch.zeros(1, 1, num_relations + 1)
    probs[0, 0, 0] = 0.30
    probs[0, 0, 1] = 0.20
    probs[0, 0, 2:13] = 0.15 / 11.0
    probs[0, 0, no_relation_idx] = 0.35
    
    # Convert probabilities to logits using log
    logits = torch.log(probs)
    
    # 1. Verify that hard argmax indeed collapses to no_relation
    hard_argmax = torch.argmax(logits, dim=-1)
    assert hard_argmax.item() == no_relation_idx, "Hard argmax should have collapsed to no_relation"
    
    # 2. Verify that confidence thresholding recovers rel_0
    preds = decode_relation_predictions(logits, no_relation_idx=no_relation_idx, confidence_threshold=0.5)
    assert preds.item() == 0, f"Expected rel_0 (index 0), got {preds.item()}"


def test_decode_relation_predictions_true_background():
    """
    Tests that true background queries (P(no_relation) > threshold) remain no_relation.
    """
    num_relations = 13
    no_relation_idx = num_relations
    
    probs = torch.zeros(1, 2, num_relations + 1)
    # Query 0: 85% background
    probs[0, 0, 0] = 0.15
    probs[0, 0, no_relation_idx] = 0.85
    
    # Query 1: 95% background
    probs[0, 1, 1] = 0.05
    probs[0, 1, no_relation_idx] = 0.95
    
    logits = torch.log(probs)
    preds = decode_relation_predictions(logits, no_relation_idx=no_relation_idx, confidence_threshold=0.5)
    
    assert preds[0, 0].item() == no_relation_idx
    assert preds[0, 1].item() == no_relation_idx


def test_decode_relation_predictions_threshold_sensitivity():
    """
    Verifies that the confidence threshold controls foreground sensitivity monotonically.
    """
    num_relations = 5
    no_relation_idx = 5
    
    probs = torch.zeros(1, 1, 6)
    probs[0, 0, 1] = 0.40  # Foreground rel_1
    probs[0, 0, no_relation_idx] = 0.60  # Background 0.60
    logits = torch.log(probs)
    
    # At threshold 0.5: no_rel_prob (0.6) > 0.5 -> background
    preds_strict = decode_relation_predictions(logits, no_relation_idx=no_relation_idx, confidence_threshold=0.5)
    assert preds_strict.item() == no_relation_idx
    
    # At threshold 0.7: no_rel_prob (0.6) <= 0.7 -> foreground (rel_1)
    preds_lenient = decode_relation_predictions(logits, no_relation_idx=no_relation_idx, confidence_threshold=0.7)
    assert preds_lenient.item() == 1


def test_compute_relation_diagnostics():
    """
    Tests probability diagnostics computation.
    """
    num_relations = 3
    no_relation_idx = 3
    
    probs = torch.tensor([
        [[0.4, 0.2, 0.1, 0.3], [0.1, 0.0, 0.1, 0.8]]
    ])
    logits = torch.log(probs)
    
    mean_fg, max_fg, mean_no_rel = compute_relation_diagnostics(logits, no_relation_idx=no_relation_idx)
    
    # Query 0 fg: 0.7, Query 1 fg: 0.2 -> mean = 0.45, max = 0.7
    # Query 0 bg: 0.3, Query 1 bg: 0.8 -> mean = 0.55
    assert abs(mean_fg - 0.45) < 1e-4
    assert abs(max_fg - 0.70) < 1e-4
    assert abs(mean_no_rel - 0.55) < 1e-4


def test_evaluate_model_on_test_diagnostics():
    """
    Tests that evaluate_model_on_test runs with confidence_threshold and returns diagnostics.
    """
    num_relations = 5
    no_relation_idx = 5
    seq_len = 16
    num_queries = 4
    num_slots = 8
    
    class MockModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            bs = input_ids.size(0)
            # Create logits where query 0 has foreground probability > 0.5
            rel_logits = torch.zeros(bs, num_queries, num_relations + 1)
            rel_logits[:, 0, 1] = 2.0  # High logit for relation 1
            rel_logits[:, 1:, no_relation_idx] = 5.0  # Background for others
            
            return {
                "rel_logits": rel_logits,
                "subj_type_logits": torch.zeros(bs, num_queries, 3),
                "obj_type_logits": torch.zeros(bs, num_queries, 3),
                "subj_slot_logits": torch.zeros(bs, num_queries, num_slots, seq_len + 1),
                "obj_slot_logits": torch.zeros(bs, num_queries, num_slots, seq_len + 1),
            }
            
    class MockTokenizer:
        def decode(self, ids, **kwargs):
            return "entity"

    dataset = {
        "input_ids": torch.randint(0, 100, (2, seq_len)),
        "attention_mask": torch.ones(2, seq_len, dtype=torch.long),
        "relations": torch.full((2, 2), no_relation_idx, dtype=torch.long),
        "subj_types": torch.zeros(2, 2, dtype=torch.long),
        "obj_types": torch.zeros(2, 2, dtype=torch.long),
        "subj_spans": torch.zeros(2, 2, 2, dtype=torch.long),
        "obj_spans": torch.zeros(2, 2, 2, dtype=torch.long),
    }

    metrics = evaluate_model_on_test(
        model=MockModel(),
        test_dataset=dataset,
        tokenizer=MockTokenizer(),
        no_relation_idx=no_relation_idx,
        device=torch.device("cpu"),
        batch_size=2,
        confidence_threshold=0.5,
    )

    assert "mean_fg_prob" in metrics
    assert "max_fg_prob" in metrics
    assert "mean_no_rel_prob" in metrics
    assert "active_query_rate" in metrics
    assert "confidence_threshold" in metrics
    assert metrics["confidence_threshold"] == 0.5
