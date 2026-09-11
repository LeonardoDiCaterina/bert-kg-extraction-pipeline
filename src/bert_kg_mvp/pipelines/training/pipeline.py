from kedro.pipeline import Pipeline, node
from .nodes import train_model
from bert_kg_mvp.pipelines.data_prep.nodes import prepare_training_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=prepare_training_data,
            inputs=["teacher_extracted_triplets", "parameters"],
            outputs=["processed_dataset", "kg_tokenizer"],
            name="prepare_data_node"
        ),
        node(
            func=train_model,
            inputs=["processed_dataset", "kg_tokenizer", "parameters"],
            outputs="trained_model",
            name="train_model_node"
        )
    ])
