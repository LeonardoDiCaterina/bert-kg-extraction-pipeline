from kedro.pipeline import Pipeline, node
from .nodes import prepare_training_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=prepare_training_data,
            inputs=["parameters"],
            outputs=["processed_dataset", "kg_tokenizer"],
            name="prepare_data_node"
        )
    ])
