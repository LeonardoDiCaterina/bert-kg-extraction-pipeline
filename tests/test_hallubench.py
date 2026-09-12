import pandas as pd
import pytest
from bert_kg_mvp.pipelines.hallubench.nodes import generate_qa, hallucination_detector, generate_hallubench_report

@pytest.fixture
def dummy_predictions():
    return pd.DataFrame({
        "doc_id": ["doc1", "doc2", "doc3"],
        "triples": ["[]", "[]", "[]"]
    })

def test_generate_qa(dummy_predictions):
    """Test that QA generation returns the correct dataframe structure."""
    qa_df = generate_qa(dummy_predictions, sample_size=2)
    assert len(qa_df) == 2
    assert "question" in qa_df.columns
    assert "answer" in qa_df.columns
    assert "grounded" in qa_df.columns

def test_hallucination_detector():
    """Test the hallucination detection logic (mocked embedding similarity)."""
    qa_df = pd.DataFrame({
        "doc_id": ["1", "2"],
        "question": ["Q1", "Q2"],
        "answer": ["A1", "A2"],
        "grounded": [1, 0]  # 1 is grounded, 0 is hallucinated
    })
    
    results = hallucination_detector(qa_df)
    
    assert len(results) == 2
    # The node assigns is_hallucination = 0 if grounded == 1
    assert results["is_hallucination"].iloc[0] == 0
    assert results["is_hallucination"].iloc[1] == 1
    assert "confidence_score" in results.columns

def test_generate_hallubench_report():
    """Test the hallucination rate math."""
    results = pd.DataFrame({
        "doc_id": ["1", "2", "3", "4"],
        "is_hallucination": [1, 1, 0, 0]  # 50% hallucination rate
    })
    
    report = generate_hallubench_report(results)
    
    assert "Total Hallucinations Detected**: 2" in report
    assert "Hallucination Rate**: 50.00%" in report
    assert "System Reliability**: 50.00%" in report

def test_generate_hallubench_report_empty():
    """Test empty dataframe handling."""
    report = generate_hallubench_report(pd.DataFrame())
    assert "No results" in report
