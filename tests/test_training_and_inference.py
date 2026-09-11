import torch
import torch.nn as nn
import pandas as pd
from unittest.mock import MagicMock, patch

from bert_kg_mvp.pipelines.inference.nodes import (
    parse_triplet_string,
    run_mvp_inference,
)
from bert_kg_mvp.pipelines.inference.pipeline import create_pipeline as create_inference_pipeline
from bert_kg_mvp.pipelines.training.nodes import train_model
from bert_kg_mvp.pipelines.training.pipeline import create_pipeline as create_training_pipeline


def test_parse_triplet_string():
    text = (
        "<triplet> Apple Inc. <subj_type> ORG <relation> Produces <obj> iPhone <obj_type> PRODUCT "
        "<triplet> Tim Cook <subj_type> PERSON <relation> Led_By <obj> Apple Inc. <obj_type> ORG"
    )
    triplets = parse_triplet_string(text)
    assert len(triplets) == 2
    assert ("Apple Inc.", "Produces", "iPhone") in triplets
    assert ("Tim Cook", "Led_By", "Apple Inc.") in triplets

    # Empty / malformed string
    assert len(parse_triplet_string("No triplets here")) == 0


class MockExtractor(nn.Module):
    def __init__(self, num_queries=3, seq_len=8, num_relations=5, num_ent_types=7, **kwargs):
        super().__init__()
        self.num_queries = num_queries
        self.seq_len = seq_len
        self.num_relations = num_relations
        self.num_ent_types = num_ent_types
        self.dummy_param = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.encoder = nn.Module()
        self.encoder.encoder = nn.Module()
        self.encoder.encoder.layer = nn.ModuleList([nn.Linear(8, 8)])

    def forward(self, input_ids, attention_mask):
        bs = input_ids.size(0)
        dev = input_ids.device
        return {
            "rel_logits": torch.randn(bs, self.num_queries, self.num_relations + 1, device=dev) * self.dummy_param,
            "subj_type_logits": torch.randn(bs, self.num_queries, self.num_ent_types, device=dev) * self.dummy_param,
            "obj_type_logits": torch.randn(bs, self.num_queries, self.num_ent_types, device=dev) * self.dummy_param,
            "subj_start_logits": torch.randn(bs, self.num_queries, self.seq_len, device=dev) * self.dummy_param,
            "subj_end_logits": torch.randn(bs, self.num_queries, self.seq_len, device=dev) * self.dummy_param,
            "obj_start_logits": torch.randn(bs, self.num_queries, self.seq_len, device=dev) * self.dummy_param,
            "obj_end_logits": torch.randn(bs, self.num_queries, self.seq_len, device=dev) * self.dummy_param,
        }


def make_dummy_dataset(n_samples=4, seq_len=8, max_triples=3):
    return {
        "input_ids": torch.randint(1, 100, (n_samples, seq_len)),
        "attention_mask": torch.ones(n_samples, seq_len, dtype=torch.int64),
        "relations": torch.zeros(n_samples, max_triples, dtype=torch.int64),
        "subj_types": torch.zeros(n_samples, max_triples, dtype=torch.int64),
        "obj_types": torch.zeros(n_samples, max_triples, dtype=torch.int64),
        "subj_spans": torch.zeros(n_samples, max_triples, 2, dtype=torch.int64),
        "obj_spans": torch.zeros(n_samples, max_triples, 2, dtype=torch.int64),
    }


def test_inference_pipeline_node():
    dataset = make_dummy_dataset(n_samples=4, seq_len=8, max_triples=3)
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "dummy entity"

    model = MockExtractor(num_queries=3, seq_len=8)
    params = {"batch_size": 2, "sample_size": 4}
    schema = {"relation_types": ["rel1", "rel2"]}

    # Test namespaced call signature
    metrics_df = run_mvp_inference(dataset, mock_tokenizer, model, params, schema)
    assert isinstance(metrics_df, pd.DataFrame)
    assert "precision" in metrics_df.columns
    assert "recall" in metrics_df.columns
    assert "f1_score" in metrics_df.columns
    assert len(metrics_df) == 1

    # Test legacy single-dict signature
    metrics_legacy = run_mvp_inference(dataset, mock_tokenizer, model, params)
    assert isinstance(metrics_legacy, pd.DataFrame)


@patch("bert_kg_mvp.pipelines.training.nodes.DynamicKGExtractor")
def test_train_model_node(mock_extractor_cls):
    mock_extractor_cls.side_effect = lambda **kwargs: MockExtractor(**kwargs)

    dataset = make_dummy_dataset(n_samples=4, seq_len=8, max_triples=3)
    mock_tokenizer = MagicMock()

    params = {
        "encoder_model_name": "bert-base-uncased",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "freeze_strategy": "partial",
        "decoder_num_layers": 1,
        "num_queries": 3,
        "gradient_accumulation_steps": 1,
    }
    schema = {
        "entity_types": ["e1", "e2"],
        "relation_types": ["r1", "r2"],
    }

    # Namespaced call
    trained_model = train_model(dataset, mock_tokenizer, params, schema)
    assert trained_model is not None

    # Single-dict call
    trained_model_legacy = train_model(dataset, mock_tokenizer, params)
    assert trained_model_legacy is not None


@patch("bert_kg_mvp.pipelines.training.nodes.DynamicKGExtractor")
def test_train_model_freeze_all(mock_extractor_cls):
    mock_extractor_cls.side_effect = lambda **kwargs: MockExtractor(**kwargs)

    dataset = make_dummy_dataset(n_samples=2, seq_len=8, max_triples=3)
    mock_tokenizer = MagicMock()

    params = {
        "encoder_model_name": "bert-base-uncased",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "freeze_strategy": "all",
        "decoder_num_layers": 1,
        "num_queries": 3,
        "gradient_accumulation_steps": 1,
    }

    trained_model = train_model(dataset, mock_tokenizer, params)
    assert trained_model is not None


@patch("bert_kg_mvp.pipelines.training.nodes.DynamicKGExtractor")
def test_train_model_with_compilation(mock_extractor_cls):
    mock_extractor_cls.side_effect = lambda **kwargs: MockExtractor(**kwargs)

    dataset = make_dummy_dataset(n_samples=2, seq_len=8, max_triples=3)
    mock_tokenizer = MagicMock()

    params = {
        "encoder_model_name": "bert-base-uncased",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "freeze_strategy": "all",
        "decoder_num_layers": 1,
        "num_queries": 3,
        "gradient_accumulation_steps": 1,
        "compile_model": True,
        "compile_mode": "default",
    }

    with patch("torch.compile", side_effect=lambda m, **kwargs: m) as mock_compile:
        trained_model = train_model(dataset, mock_tokenizer, params)
        assert trained_model is not None
        mock_compile.assert_called_once()


def test_pipeline_definitions():
    training_pipe = create_training_pipeline()
    assert "train_model_node" in [n.name for n in training_pipe.nodes]
    assert "prepare_data_node" in [n.name for n in training_pipe.nodes]

    inference_pipe = create_inference_pipeline()
    assert "run_inference_node" in [n.name for n in inference_pipe.nodes]

