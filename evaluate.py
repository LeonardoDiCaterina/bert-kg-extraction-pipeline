from pathlib import Path
import pandas as pd
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession

from bert_kg_mvp.pipelines.inference.nodes import run_mvp_inference
from bert_kg_mvp.utils import parse_triplet_string

__all__ = ["parse_triplet_string", "evaluate_baseline"]


def evaluate_baseline() -> pd.DataFrame:
    """Loads Kedro catalog and context to evaluate the trained model on validation data."""
    bootstrap_project(Path.cwd())
    with KedroSession.create() as session:
        context = session.load_context()
        tokenizer = context.catalog.load("kg_tokenizer")
        model = context.catalog.load("trained_model")
        dataset = context.catalog.load("processed_dataset")
        params = context.params

    inference_params = params.get("inference", {})
    schema_params = params.get("schema", {})

    return run_mvp_inference(
        processed_dataset=dataset,
        tokenizer=tokenizer,
        trained_model=model,
        inference_params=inference_params,
        schema_params=schema_params,
    )


if __name__ == "__main__":
    evaluate_baseline()
