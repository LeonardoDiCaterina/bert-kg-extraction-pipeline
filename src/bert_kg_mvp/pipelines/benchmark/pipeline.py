from kedro.pipeline import Pipeline, node, pipeline
from .nodes import run_encoder_benchmark, split_dataset_node


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates the Kedro pipeline for company-stratified multi-encoder benchmarking.
    """
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
                outputs="benchmark_results_table",
                name="run_encoder_benchmark_node",
            ),
        ]
    )
