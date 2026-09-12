import pandas as pd
import pytest
from bert_kg_mvp.pipelines.evalbench.nodes import sample_predictions, llm_judge_evaluation, generate_evalbench_report

@pytest.fixture
def dummy_predictions():
    return pd.DataFrame({
        "doc_id": ["doc1", "doc2", "doc3", "doc4", "doc5"],
        "triples": ["[]", "[]", "[]", "[]", "[]"]
    })

def test_sample_predictions(dummy_predictions):
    """Test that sampling returns the correct number of rows."""
    sample = sample_predictions(dummy_predictions, sample_size=2)
    assert len(sample) == 2
    
def test_sample_predictions_oversample(dummy_predictions):
    """Test that sampling handles requests larger than the dataset gracefully."""
    sample = sample_predictions(dummy_predictions, sample_size=10)
    assert len(sample) == 5

def test_llm_judge_evaluation(dummy_predictions):
    """
    Test the judge evaluation node. 
    Because the node relies on a mocked/placeholder LLM loop internally,
    this test does not require spinning up a real LLM endpoint.
    """
    judgments = llm_judge_evaluation(dummy_predictions, judge_models=["Qwen-72B"])
    
    assert len(judgments) == 5
    assert list(judgments.columns) == ["doc_id", "faithfulness", "precision", "relevance", "comprehensiveness"]
    assert judgments["faithfulness"].iloc[0] == 1

def test_generate_evalbench_report():
    """Test the formatting and math of the markdown report generation."""
    judgments = pd.DataFrame({
        "doc_id": ["1", "2"],
        "faithfulness": [1, 0],      # 50%
        "precision": [1, 1],         # 100%
        "relevance": [0, 0],         # 0%
        "comprehensiveness": [3, 1]  # 4 / 6 = 66.66%
    })
    
    report = generate_evalbench_report(judgments)
    
    assert "Faithfulness**: 50.00%" in report
    assert "Precision**: 100.00%" in report
    assert "Relevance**: 0.00%" in report
    assert "Comprehensiveness**: 66.67%" in report

def test_generate_evalbench_report_empty():
    """Test empty dataframe handling."""
    report = generate_evalbench_report(pd.DataFrame())
    assert "No judgments" in report
