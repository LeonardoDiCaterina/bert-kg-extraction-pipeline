import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from bert_kg_mvp.pipelines.data_prep.teacher_node import (
    get_company_name,
    format_schema_prompt,
    generate_teacher_triplets,
)


def test_get_company_name():
    assert get_company_name("AAPL_2024.pdf") == "Apple Inc."
    assert get_company_name("MSFT_10k_submission") == "Microsoft Corp."
    assert get_company_name("AMZN_filing") == "Amazon.com, Inc."
    assert get_company_name("XYZ_10k.txt") == "The Corporation"


def test_format_schema_prompt():
    # Empty schema fallback
    assert "Entity Types: ORG" in format_schema_prompt(None)

    # Custom schema
    custom = {
        "entity_types": ["org", "fin_metric"],
        "relation_types": ["has_metric", "reports_risk"]
    }
    rendered = format_schema_prompt(custom)
    assert "ORG, FIN_METRIC" in rendered
    assert "Has_Metric, Reports_Risk" in rendered


def test_generate_teacher_triplets_missing_vllm():
    with patch("bert_kg_mvp.pipelines.data_prep.teacher_node.LLM", None):
        with pytest.raises(ImportError, match="vllm is required"):
            df = pd.DataFrame([{"doc_id": "AAPL_1", "chunk_id": 1, "text": "sample"}])
            generate_teacher_triplets(df)


def test_generate_teacher_triplets_mocked_llm():
    mock_llm_instance = MagicMock()

    class MockOutput:
        def __init__(self, text):
            self.outputs = [MagicMock(text=text)]

    mock_llm_instance.generate.side_effect = [
        # Agent 1 outputs
        [MockOutput('[{"head": "Apple Inc.", "head_type": "ORG", "relation": "Produces", "tail": "iPhone", "tail_type": "PRODUCT"}]')],
        # Agent 2 outputs
        [MockOutput("PASS")],
        # Agent 3 outputs (with markdown wrapper)
        [MockOutput('```json\n[{"head": "Apple Inc.", "head_type": "ORG", "relation": "Produces", "tail": "iPhone", "tail_type": "PRODUCT"}]\n```')],
    ]

    mock_llm_cls = MagicMock(return_value=mock_llm_instance)
    mock_sampling_params = MagicMock()

    with patch("bert_kg_mvp.pipelines.data_prep.teacher_node.LLM", mock_llm_cls), \
         patch("bert_kg_mvp.pipelines.data_prep.teacher_node.SamplingParams", mock_sampling_params):
        df_input = pd.DataFrame([{
            "doc_id": "AAPL_2024.pdf",
            "chunk_id": "chunk_0",
            "text": "Apple produces iPhone."
        }])

        teacher_params = {
            "model_name": "mock_qwen",
            "temperature": 0.2,
            "max_tokens": 512,
            "company_map": {"AAPL": "Apple Inc."}
        }
        schema_params = {
            "entity_types": ["org", "product"],
            "relation_types": ["produces"]
        }

        result_df = generate_teacher_triplets(df_input, teacher_params=teacher_params, schema_params=schema_params)
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 1
        assert "triples" in result_df.columns
        assert len(result_df.iloc[0]["triples"]) == 1
        assert result_df.iloc[0]["triples"][0]["head"] == "Apple Inc."


def test_generate_teacher_triplets_checkpoint_resumption(tmp_path):
    """Verifies that existing checkpoints skip Agent 1 and Agent 2 generation."""
    agent1_file = tmp_path / "teacher_agent1_extractions.parquet"
    agent2_file = tmp_path / "teacher_agent2_critiques.parquet"

    pd.DataFrame({"output": ['[{"head": "Apple Inc.", "head_type": "ORG", "relation": "Produces", "tail": "Mac", "tail_type": "PRODUCT"}]']}).to_parquet(agent1_file)
    pd.DataFrame({"output": ["PASS"]}).to_parquet(agent2_file)

    mock_llm_instance = MagicMock()

    class MockOutput:
        def __init__(self, text):
            self.outputs = [MagicMock(text=text)]

    # Only Agent 3 should be invoked because Agent 1 and Agent 2 checkpoints exist!
    mock_llm_instance.generate.return_value = [
        MockOutput('[{"head": "Apple Inc.", "head_type": "ORG", "relation": "Produces", "tail": "Mac", "tail_type": "PRODUCT"}]')
    ]

    mock_llm_cls = MagicMock(return_value=mock_llm_instance)
    mock_sampling_params = MagicMock()

    with patch("bert_kg_mvp.pipelines.data_prep.teacher_node.LLM", mock_llm_cls), \
         patch("bert_kg_mvp.pipelines.data_prep.teacher_node.SamplingParams", mock_sampling_params):
        df_input = pd.DataFrame([{
            "doc_id": "AAPL_2024.pdf",
            "chunk_id": "chunk_0",
            "text": "Apple produces Mac."
        }])

        teacher_params = {
            "checkpoint_dir": str(tmp_path),
            "resume_checkpoints": True,
            "company_map": {"AAPL": "Apple Inc."},
        }

        result_df = generate_teacher_triplets(df_input, teacher_params=teacher_params)
        # Verify generate was called exactly ONCE (for Agent 3)
        assert mock_llm_instance.generate.call_count == 1
        assert len(result_df) == 1
        assert result_df.iloc[0]["triples"][0]["tail"] == "Mac"


def test_generate_teacher_triplets_subsampling():
    """Verifies that max_samples properly truncates and samples the dataset."""
    mock_llm_instance = MagicMock()

    class MockOutput:
        def __init__(self, text):
            self.outputs = [MagicMock(text=text)]

    mock_llm_instance.generate.return_value = [MockOutput("[]")] * 2

    mock_llm_cls = MagicMock(return_value=mock_llm_instance)
    mock_sampling_params = MagicMock()

    with patch("bert_kg_mvp.pipelines.data_prep.teacher_node.LLM", mock_llm_cls), \
         patch("bert_kg_mvp.pipelines.data_prep.teacher_node.SamplingParams", mock_sampling_params):
        df_input = pd.DataFrame([
            {"doc_id": "AAPL_2024.pdf", "ticker": "AAPL", "chunk_id": f"chunk_{i}", "text": f"Text {i}"}
            for i in range(10)
        ])

        teacher_params = {
            "max_samples": 2,
            "checkpoint_dir": None,
            "resume_checkpoints": False,
        }

        result_df = generate_teacher_triplets(df_input, teacher_params=teacher_params)
        assert len(result_df) == 2


def test_generate_teacher_triplets_prompt_clamping(tmp_path):
    """Verifies that text, triples, and critique are clamped defensively in Agent 3."""
    mock_llm_instance = MagicMock()

    captured_prompts = []

    def mock_generate(prompts, sampling_params):
        captured_prompts.append(prompts)
        class MockOutput:
            def __init__(self, text):
                self.outputs = [MagicMock(text=text)]
        return [MockOutput("[]")] * len(prompts)

    mock_llm_instance.generate.side_effect = mock_generate
    mock_llm_cls = MagicMock(return_value=mock_llm_instance)
    mock_sampling_params = MagicMock()

    with patch("bert_kg_mvp.pipelines.data_prep.teacher_node.LLM", mock_llm_cls), \
         patch("bert_kg_mvp.pipelines.data_prep.teacher_node.SamplingParams", mock_sampling_params):
        df_input = pd.DataFrame([{
            "doc_id": "AAPL_2024.pdf",
            "chunk_id": "chunk_0",
            "text": "A" * 10000
        }])

        teacher_params = {
            "checkpoint_dir": str(tmp_path),
            "resume_checkpoints": False,
            "max_refiner_text_chars": 50,
            "max_refiner_triples_chars": 50,
            "max_refiner_critique_chars": 50,
        }

        generate_teacher_triplets(df_input, teacher_params=teacher_params)
        assert len(captured_prompts) == 3
        ref_prompt = captured_prompts[2][0]
        # Verify text was clamped to 50 chars in Agent 3 refiner prompt
        assert "A" * 51 not in ref_prompt
        assert "A" * 50 in ref_prompt
