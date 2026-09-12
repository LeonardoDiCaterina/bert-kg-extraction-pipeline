import torch
from bert_kg_mvp.models.bipartite_loss import SetCriterion


def make_dummy_outputs(
    batch_size=2, num_queries=5, seq_len=16, num_relations=3, num_entity_types=4
):
    return {
        "rel_logits": torch.randn(
            batch_size, num_queries, num_relations + 1, requires_grad=True
        ),
        "subj_type_logits": torch.randn(
            batch_size, num_queries, num_entity_types, requires_grad=True
        ),
        "obj_type_logits": torch.randn(
            batch_size, num_queries, num_entity_types, requires_grad=True
        ),
        "subj_start_logits": torch.randn(
            batch_size, num_queries, seq_len, requires_grad=True
        ),
        "subj_end_logits": torch.randn(
            batch_size, num_queries, seq_len, requires_grad=True
        ),
        "obj_start_logits": torch.randn(
            batch_size, num_queries, seq_len, requires_grad=True
        ),
        "obj_end_logits": torch.randn(
            batch_size, num_queries, seq_len, requires_grad=True
        ),
    }


def test_set_criterion_init():
    criterion = SetCriterion(num_relation_classes=4, num_entity_types=5, eos_coef=0.2)
    assert criterion.num_relation_classes == 4
    assert criterion.num_entity_types == 5
    assert criterion.empty_weight.shape == (5,)
    assert criterion.empty_weight[-1] == 0.2


def test_set_criterion_forward_with_targets():
    criterion = SetCriterion(num_relation_classes=3, num_entity_types=4)
    outputs = make_dummy_outputs()

    targets = [
        {
            "relations": torch.tensor([0, 1], dtype=torch.int64),
            "subj_types": torch.tensor([0, 2], dtype=torch.int64),
            "obj_types": torch.tensor([1, 3], dtype=torch.int64),
            "subj_spans": torch.tensor([[1, 3], [4, 6]], dtype=torch.int64),
            "obj_spans": torch.tensor([[2, 4], [7, 9]], dtype=torch.int64),
        },
        {
            "relations": torch.tensor([2], dtype=torch.int64),
            "subj_types": torch.tensor([1], dtype=torch.int64),
            "obj_types": torch.tensor([0], dtype=torch.int64),
            "subj_spans": torch.tensor([[0, 2]], dtype=torch.int64),
            "obj_spans": torch.tensor([[3, 5]], dtype=torch.int64),
        },
    ]

    losses = criterion(outputs, targets)
    assert "loss_ce" in losses
    assert "loss_type" in losses
    assert "loss_span" in losses

    total_loss = sum(losses.values())
    assert total_loss.item() > 0
    total_loss.backward()
    assert outputs["rel_logits"].grad is not None


def test_set_criterion_empty_targets():
    criterion = SetCriterion(num_relation_classes=3, num_entity_types=4)
    outputs = make_dummy_outputs()

    targets = [
        {
            "relations": torch.tensor([], dtype=torch.int64),
            "subj_types": torch.tensor([], dtype=torch.int64),
            "obj_types": torch.tensor([], dtype=torch.int64),
            "subj_spans": torch.empty((0, 2), dtype=torch.int64),
            "obj_spans": torch.empty((0, 2), dtype=torch.int64),
        },
        {
            "relations": torch.tensor([], dtype=torch.int64),
            "subj_types": torch.tensor([], dtype=torch.int64),
            "obj_types": torch.tensor([], dtype=torch.int64),
            "subj_spans": torch.empty((0, 2), dtype=torch.int64),
            "obj_spans": torch.empty((0, 2), dtype=torch.int64),
        },
    ]

    losses = criterion(outputs, targets)
    assert "loss_ce" in losses
    assert "loss_type" not in losses
    assert "loss_span" not in losses
    assert losses["loss_ce"].item() > 0
