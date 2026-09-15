from kedro.pipeline import Pipeline, node
from bert_kg_mvp.pipelines.data_prep.nodes import prepare_training_data
from bert_kg_mvp.pipelines.benchmark.nodes import split_dataset_node
from .nodes import train_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_dataset_node,
                inputs=["teacher_extracted_triplets", "params:split"],
                outputs="split_triplets_data",
                name="split_triplets_node",
            ),
            node(
                func=lambda splits: splits["train"],
                inputs="split_triplets_data",
                outputs="train_triplets",
                name="get_train_split",
            ),
            node(
                func=lambda splits: splits["val"],
                inputs="split_triplets_data",
                outputs="val_triplets",
                name="get_val_split",
            ),
            node(
                func=prepare_training_data,
                inputs=[
                    "train_triplets",
                    "params:data_prep",
                    "params:training",
                    "params:schema",
                ],
                outputs=["processed_dataset", "kg_tokenizer"],
                name="prepare_data_node",
            ),
            node(
                func=prepare_training_data,
                inputs=[
                    "val_triplets",
                    "params:data_prep",
                    "params:training",
                    "params:schema",
                ],
                outputs=["val_processed_dataset", "val_kg_tokenizer"],
                name="prepare_val_data_node",
            ),
            node(
                func=train_model,
                inputs=[
                    "processed_dataset",
                    "kg_tokenizer",
                    "params:training",
                    "params:schema",
                    "val_processed_dataset",
                ],
                outputs="trained_model",
                name="train_model_node",
            ),
        ]
    )
