from unittest.mock import MagicMock, patch
import pandas as pd
import torch

from bert_kg_mvp.pipelines.benchmark.nodes import (
    evaluate_model_on_test,
    run_encoder_benchmark,
    split_dataset_node,
)


def test_split_dataset_node():
    data = pd.DataFrame([
        {"doc_id": "AAPL_1.pdf", "text": "text 1", "triples": "[]"},
        {"doc_id": "AAPL_2.pdf", "text": "text 2", "triples": "[]"},
        {"doc_id": "MSFT_1.pdf", "text": "text 3", "triples": "[]"},
        {"doc_id": "NVDA_1.pdf", "text": "text 4", "triples": "[]"},
    ])
    splits = split_dataset_node(data, split_params={"strategy": "company_stratified", "random_seed": 42})
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(data)


def test_evaluate_model_on_test_mock():
    # Mock model and test tensors
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.return_value = {
        "rel_logits": torch.zeros((1, 2, 6)),
        "subj_start_logits": torch.zeros((1, 2, 10)),
        "subj_end_logits": torch.zeros((1, 2, 10)),
        "obj_start_logits": torch.zeros((1, 2, 10)),
        "obj_end_logits": torch.zeros((1, 2, 10)),
    }

    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "entity"

    test_ds = {
        "input_ids": torch.zeros((1, 10), dtype=torch.long),
        "attention_mask": torch.ones((1, 10), dtype=torch.long),
        "relations": torch.full((1, 2), 5, dtype=torch.long),
        "subj_spans": torch.zeros((1, 2, 2), dtype=torch.long),
        "obj_spans": torch.zeros((1, 2, 2), dtype=torch.long),
    }

    prec, rec, f1, latency = evaluate_model_on_test(
        model=mock_model,
        test_dataset=test_ds,
        tokenizer=mock_tokenizer,
        no_relation_idx=5,
        device=torch.device("cpu"),
        batch_size=1,
    )
    assert 0.0 <= prec <= 1.0
    assert 0.0 <= rec <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert latency >= 0.0


@patch("bert_kg_mvp.pipelines.benchmark.nodes.prepare_training_data")
@patch("bert_kg_mvp.pipelines.benchmark.nodes.DynamicKGExtractor")
def test_run_encoder_benchmark_mock(mock_model_cls, mock_prep_data):
    mock_prep_data.return_value = (
        {
            "input_ids": torch.zeros((2, 8), dtype=torch.long),
            "attention_mask": torch.ones((2, 8), dtype=torch.long),
            "relations": torch.full((2, 2), 5, dtype=torch.long),
            "subj_types": torch.zeros((2, 2), dtype=torch.long),
            "obj_types": torch.zeros((2, 2), dtype=torch.long),
            "subj_spans": torch.zeros((2, 2, 2), dtype=torch.long),
            "obj_spans": torch.zeros((2, 2, 2), dtype=torch.long),
        },
        MagicMock(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    mock_instance = MagicMock()
    mock_param = torch.nn.Parameter(torch.zeros(2, 2, device=device))
    mock_instance.parameters.return_value = [mock_param]
    mock_instance.to.return_value = mock_instance
    mock_instance.side_effect = lambda ids, mask: {
        "rel_logits": torch.zeros((ids.size(0), 2, 6), device=ids.device, requires_grad=True),
        "subj_start_logits": torch.zeros((ids.size(0), 2, 8), device=ids.device),
        "subj_end_logits": torch.zeros((ids.size(0), 2, 8), device=ids.device),
        "obj_start_logits": torch.zeros((ids.size(0), 2, 8), device=ids.device),
        "obj_end_logits": torch.zeros((ids.size(0), 2, 8), device=ids.device),
    }
    mock_model_cls.return_value = mock_instance

    split_data = {
        "train": pd.DataFrame([{"doc_id": "AAPL", "text": "t", "triples": "[]"}]),
        "val": pd.DataFrame([{"doc_id": "MSFT", "text": "t", "triples": "[]"}]),
        "test": pd.DataFrame([{"doc_id": "NVDA", "text": "t", "triples": "[]"}]),
    }

    benchmark_params = {
        "models": [{"name": "bert-base-uncased", "display_name": "BERT Base"}],
        "epochs": 1,
        "batch_size": 2,
    }

    results_df = run_encoder_benchmark(
        split_data=split_data,
        benchmark_params=benchmark_params,
        data_prep_params={},
        schema_params={},
    )
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == 1
    assert "model_name" in results_df.columns
    assert "test_f1" in results_df.columns
