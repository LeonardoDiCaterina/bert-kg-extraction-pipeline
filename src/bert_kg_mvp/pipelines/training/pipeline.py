from kedro.pipeline import Pipeline, node
from .nodes import train_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=train_model,
            inputs=["processed_dataset", "kg_tokenizer", "parameters"],
            outputs="trained_model",
            name="train_model_node"
        )
    ])
