import pytest
import pandas as pd
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from bert_kg_mvp.pipelines.benchmark.nodes import (
    split_dataset_node,
    run_encoder_benchmark,
    run_decoder_benchmark
)

def test_split_dataset_node():
    data = [
        {"doc_id": "A_1", "triples": "1"},
        {"doc_id": "A_2", "triples": "2"},
        {"doc_id": "B_1", "triples": "3"},
        {"doc_id": "C_1", "triples": "4"},
        {"doc_id": "C_2", "triples": "5"},
        {"doc_id": "D_1", "triples": "6"},
    ]
    df = pd.DataFrame(data)

    params = {
        "strategy": "company_stratified",
        "train_ratio": 0.5,
        "val_ratio": 0.25,
        "test_ratio": 0.25,
        "random_seed": 42,
    }
    splits = split_dataset_node(df, params)

    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

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

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    mock_instance = MagicMock()
    mock_param = torch.nn.Parameter(torch.zeros(2, 2, device=device))
    mock_instance.parameters.return_value = [mock_param]
    mock_instance.to.return_value = mock_instance
    mock_instance.side_effect = lambda ids, mask: {
        "rel_logits": torch.zeros((ids.size(0), 2, 6), device=ids.device, requires_grad=True),
        "subj_type_logits": torch.zeros((ids.size(0), 2, 6), device=ids.device),
        "obj_type_logits": torch.zeros((ids.size(0), 2, 6), device=ids.device),
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
    assert "test_strict_f1" in results_df.columns

@patch("bert_kg_mvp.pipelines.benchmark.nodes.prepare_training_data")
@patch("bert_kg_mvp.pipelines.benchmark.nodes.build_decoder")
def test_run_decoder_benchmark_mock(mock_build_decoder, mock_prep_data):
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

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    mock_instance = MagicMock()
    mock_param = torch.nn.Parameter(torch.zeros(2, 2, device=device))
    mock_instance.parameters.return_value = [mock_param]
    mock_instance.to.return_value = mock_instance
    mock_instance.side_effect = lambda ids, mask: {
        "rel_logits": torch.zeros((ids.size(0), 2, 6), device=ids.device, requires_grad=True),
        "subj_type_logits": torch.zeros((ids.size(0), 2, 6), device=ids.device),
        "obj_type_logits": torch.zeros((ids.size(0), 2, 6), device=ids.device),
        "subj_start_logits": torch.zeros((ids.size(0), 2, 8), device=ids.device),
        "subj_end_logits": torch.zeros((ids.size(0), 2, 8), device=ids.device),
        "obj_start_logits": torch.zeros((ids.size(0), 2, 8), device=ids.device),
        "obj_end_logits": torch.zeros((ids.size(0), 2, 8), device=ids.device),
    }
    mock_build_decoder.return_value = mock_instance

    split_data = {
        "train": pd.DataFrame([{"doc_id": "AAPL", "text": "t", "triples": "[]"}]),
        "val": pd.DataFrame([{"doc_id": "MSFT", "text": "t", "triples": "[]"}]),
        "test": pd.DataFrame([{"doc_id": "NVDA", "text": "t", "triples": "[]"}]),
    }

    benchmark_params = {
        "decoder_configs": [{"decoder_type": "baseline"}],
        "epochs": 1,
        "batch_size": 2,
    }

    results_df = run_decoder_benchmark(
        split_data=split_data,
        benchmark_params=benchmark_params,
        data_prep_params={},
        schema_params={},
    )
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == 1
    assert "decoder_type" in results_df.columns
    assert "test_strict_f1" in results_df.columns
