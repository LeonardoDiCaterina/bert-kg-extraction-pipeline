from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_evaluation_report
from bert_kg_mvp.pipelines.benchmark.nodes import split_dataset_node

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_dataset_node,
                inputs=["teacher_extracted_triplets", "params:split"],
                outputs="split_triplets_data",
                name="split_triplets_node_reporting",
            ),
            node(
                func=generate_evaluation_report,
                inputs=[
                    "split_triplets_data",
                    "params:reporting",
                    "params:data_prep",
                    "params:schema"
                ],
                outputs=["evaluation_report_md", "knowledge_graph_png"],
                name="generate_evaluation_report_node",
            )
        ]
    )
