from kedro.pipeline import Pipeline, node
from bert_kg_mvp.pipelines.data_prep.nodes import prepare_training_data
from .nodes import train_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=prepare_training_data,
                inputs=[
                    "teacher_extracted_triplets",
                    "params:data_prep",
                    "params:training",
                    "params:schema",
                ],
                outputs=["processed_dataset", "kg_tokenizer"],
                name="prepare_data_node",
            ),
            node(
                func=train_model,
                inputs=[
                    "processed_dataset",
                    "kg_tokenizer",
                    "params:training",
                    "params:schema",
                ],
                outputs="trained_model",
                name="train_model_node",
            ),
        ]
    )
