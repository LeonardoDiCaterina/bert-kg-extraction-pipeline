import os
import torch
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from bert_kg_mvp.models import build_decoder
from bert_kg_mvp.pipelines.reporting.nodes import generate_evaluation_report


def test_save_model_state_dict(tmp_path):
    """
    Instantiates the architecture with random weights and saves its state_dict 
    to verify that PyTorch native saving works correctly.
    """
    model = build_decoder(
        decoder_type="baseline",
        encoder_model_name="bert-base-uncased",
        num_relations=5,
        num_ent_types=7,
        num_queries=15,
        num_layers=2,
        d_model=768,
        freeze_strategy="partial",
        unfrozen_top_layers=2,
    )
    
    save_path = tmp_path / "test_model_weights.pt"
    
    # Save only the state_dict (which is what train_model now returns)
    state_dict = getattr(model, "_orig_mod", model).state_dict()
    torch.save({"model_state_dict": state_dict}, save_path)
    
    assert os.path.exists(save_path)
    
    # Verify it can be loaded
    loaded_checkpoint = torch.load(save_path, weights_only=False)
    assert "model_state_dict" in loaded_checkpoint
    
    # Verify loading back into model works
    model.load_state_dict(loaded_checkpoint["model_state_dict"])


@patch("bert_kg_mvp.pipelines.reporting.nodes.prepare_training_data")
@patch("bert_kg_mvp.pipelines.reporting.nodes.evaluate_model_on_test")
def test_reporting_pipeline_with_cached_weights(mock_eval, mock_prep, tmp_path):
    """
    Takes the saved weights from the previous test and runs the reporting 
    pipeline on it to ensure it successfully loads the state_dict instead of 
    expecting a raw pickle object.
    """
    # 1. Create a model and save its state_dict
    model = build_decoder(
        decoder_type="baseline",
        encoder_model_name="bert-base-uncased",
        num_relations=5,
        num_ent_types=7,
        num_queries=15,
        num_layers=4,
        d_model=768,
        freeze_strategy="partial",
        unfrozen_top_layers=4,
    )
    save_path = tmp_path / "test_best_model.pt"
    
    state_dict = getattr(model, "_orig_mod", model).state_dict()
    torch.save({"model_state_dict": state_dict}, save_path)
    
    # 2. Mock the data preparation and evaluation to isolate the loading logic
    mock_prep.return_value = (
        {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "attention_mask": torch.ones((2, 8)),
            "relations": torch.zeros((2, 2)),
            "subj_types": torch.zeros((2, 2)),
            "obj_types": torch.zeros((2, 2)),
            "subj_spans": torch.zeros((2, 2, 2), dtype=torch.long),
            "obj_spans": torch.zeros((2, 2, 2), dtype=torch.long),
        },
        {"strict_f1": 0.5, "type_accuracy": 0.8}
    )
    
    mock_eval.return_value = {
        "strict_f1": 0.85,
        "span_f1": 0.90,
        "type_accuracy": 0.88,
        "rel_accuracy": 0.82,
        "mean_error_distance": 1.2,
        "jaccard_mean": 0.95,
        "jaccard_median": 1.0,
    }
    
    split_data = {
        "test": pd.DataFrame([{"doc_id": "test_1", "text": "dummy", "triples": "dummy"}])
    }
    
    reporting_params = {
        "checkpoint_path": str(save_path),
        "encoder_name": "bert-base-uncased",
        "num_examples_to_print": 1,
        "visualize_chunk_index": 0,
    }
    
    data_prep_params = {"max_seq_length": 8}
    schema_params = {
        "relation_types": ["rel1", "rel2", "rel3", "rel4", "rel5"],
        "entity_types": ["ent1", "ent2", "ent3", "ent4", "ent5", "ent6", "ent7"],
    }
    
    # 3. Run the reporting node
    markdown_report, fig = generate_evaluation_report(
        split_data=split_data,
        reporting_params=reporting_params,
        data_prep_params=data_prep_params,
        schema_params=schema_params,
    )
    
    # 4. Verify it succeeded
    assert markdown_report is not None
    assert "Model Evaluation Report" in markdown_report
    assert fig is not None
