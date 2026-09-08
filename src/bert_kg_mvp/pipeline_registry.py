from typing import Dict
from kedro.pipeline import Pipeline
from bert_kg_mvp.pipelines import data_prep, inference, training

def register_pipelines() -> Dict[str, Pipeline]:
    data_prep_pipeline = data_prep.create_pipeline()
    inference_pipeline = inference.create_pipeline()
    training_pipeline = training.create_pipeline()

    return {
        "__default__": data_prep_pipeline + training_pipeline + inference_pipeline,
        "data_prep": data_prep_pipeline,
        "training": training_pipeline,
        "inference": inference_pipeline,
    }
