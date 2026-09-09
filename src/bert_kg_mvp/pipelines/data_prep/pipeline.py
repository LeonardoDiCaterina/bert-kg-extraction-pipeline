from bert_kg_mvp.pipelines.data_prep.nodes import parse_sec_filings, prepare_training_data
from kedro.pipeline import Pipeline, node
from .teacher_node import generate_teacher_triplets

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=parse_sec_filings,
            inputs=["params:raw_pdf_dir", "params:max_chunk_words"],
            outputs="parsed_10k_chunks",
            name="parse_sec_filings_node",
        ),
        node(
            func=generate_teacher_triplets,
            inputs="parsed_10k_chunks",
            outputs="teacher_extracted_triplets",
            name="generate_teacher_triplets_node"
        ),
        node(
            func=prepare_training_data,
            inputs=["teacher_extracted_triplets", "parameters"],
            outputs=["processed_dataset", "kg_tokenizer"],
            name="prepare_data_node"
        )   
    ])
