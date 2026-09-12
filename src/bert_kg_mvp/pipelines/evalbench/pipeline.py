from kedro.pipeline import Pipeline, node, pipeline
from .nodes import sample_predictions, llm_judge_evaluation, generate_evalbench_report

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=sample_predictions,
                inputs=["student_predictions", "params:evalbench.sample_size"],
                outputs="evalbench_sample",
                name="sample_predictions_node",
            ),
            node(
                func=llm_judge_evaluation,
                inputs=["evalbench_sample", "params:evalbench.judge_models"],
                outputs="evalbench_judgments",
                name="llm_judge_evaluation_node",
            ),
            node(
                func=generate_evalbench_report,
                inputs="evalbench_judgments",
                outputs="evalbench_report",
                name="generate_evalbench_report_node",
            ),
        ]
    )
