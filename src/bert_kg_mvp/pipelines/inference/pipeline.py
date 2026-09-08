from kedro.pipeline import Pipeline, node
from .nodes import run_mvp_inference

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=run_mvp_inference,
            inputs=["processed_dataset", "kg_tokenizer", "trained_model", "parameters"],
            outputs="kg_inference_output",
            name="run_inference_node"
        )
    ])
