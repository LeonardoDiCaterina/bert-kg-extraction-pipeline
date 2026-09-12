from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_qa, hallucination_detector, generate_hallubench_report

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_qa,
                inputs=["student_predictions", "params:hallubench.sample_size"],
                outputs="hallubench_qa_dataset",
                name="generate_qa_node",
            ),
            node(
                func=hallucination_detector,
                inputs="hallubench_qa_dataset",
                outputs="hallubench_results",
                name="hallucination_detector_node",
            ),
            node(
                func=generate_hallubench_report,
                inputs="hallubench_results",
                outputs="hallubench_report",
                name="generate_hallubench_report_node",
            ),
        ]
    )
