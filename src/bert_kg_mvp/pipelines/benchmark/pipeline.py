from kedro.pipeline import Pipeline, node, pipeline
from .nodes import run_encoder_benchmark, run_decoder_benchmark, split_dataset_node

def create_encoder_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_dataset_node,
                inputs=["teacher_extracted_triplets", "params:split"],
                outputs="split_triplets_data",
                name="split_triplets_node",
            ),
            node(
                func=run_encoder_benchmark,
                inputs=[
                    "split_triplets_data",
                    "params:benchmark",
                    "params:data_prep",
                    "params:schema",
                ],
                outputs="encoder_benchmark_results_table",
                name="run_encoder_benchmark_node",
            ),
        ]
    )

def create_decoder_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_dataset_node,
                inputs=["teacher_extracted_triplets", "params:split"],
                outputs="split_triplets_data",
                name="split_triplets_node_decoder",
            ),
            node(
                func=run_decoder_benchmark,
                inputs=[
                    "split_triplets_data",
                    "params:benchmark",
                    "params:data_prep",
                    "params:schema",
                ],
                outputs="decoder_benchmark_results_table",
                name="run_decoder_benchmark_node",
            ),
        ]
    )
