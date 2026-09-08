from kedro.pipeline import Pipeline, node
from .nodes import prepare_training_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=prepare_training_data,
            inputs=["raw_texts", "parameters"],
            outputs=["processed_dataset", "kg_tokenizer"],
            name="prepare_training_data_node"
        )
    ])
