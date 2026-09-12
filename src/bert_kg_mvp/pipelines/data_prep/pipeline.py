from bert_kg_mvp.pipelines.data_prep.nodes import parse_sec_filings
from kedro.pipeline import Pipeline, node
from .teacher_node import generate_teacher_triplets


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=parse_sec_filings,
                inputs="params:data_prep",
                outputs="parsed_10k_chunks",
                name="parse_sec_filings_node",
            ),
            node(
                func=generate_teacher_triplets,
                inputs=["parsed_10k_chunks", "params:teacher", "params:schema"],
                outputs="teacher_extracted_triplets",
                name="generate_teacher_triplets_node",
            ),
        ]
    )
