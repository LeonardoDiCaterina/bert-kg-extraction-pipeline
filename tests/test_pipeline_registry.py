from kedro.pipeline import Pipeline
from bert_kg_mvp.pipeline_registry import register_pipelines


def test_register_pipelines():
    pipelines = register_pipelines()
    assert isinstance(pipelines, dict)
    assert "__default__" in pipelines
    assert "data_prep" in pipelines
    assert "training" in pipelines
    assert "inference" in pipelines
    assert "benchmark" in pipelines

    for name, pipe in pipelines.items():
        assert isinstance(pipe, Pipeline)

