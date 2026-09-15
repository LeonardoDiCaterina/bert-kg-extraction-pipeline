from typing import Dict
from kedro.pipeline import Pipeline
from bert_kg_mvp.pipelines import benchmark, data_prep, inference, training, evalbench, hallubench, reporting


def register_pipelines() -> Dict[str, Pipeline]:
    data_prep_pipeline = data_prep.create_pipeline()
    inference_pipeline = inference.create_pipeline()
    training_pipeline = training.create_pipeline()
    benchmark_encoders_pipeline = benchmark.create_encoder_pipeline()
    benchmark_decoders_pipeline = benchmark.create_decoder_pipeline()
    evalbench_pipeline = evalbench.create_pipeline()
    hallubench_pipeline = hallubench.create_pipeline()
    reporting_pipeline = reporting.create_pipeline()

    return {
        "__default__": data_prep_pipeline + training_pipeline + inference_pipeline,
        "data_prep": data_prep_pipeline,
        "training": training_pipeline,
        "inference": inference_pipeline,
        "benchmark_encoders": benchmark_encoders_pipeline,
        "benchmark_decoders": benchmark_decoders_pipeline,
        "evalbench": evalbench_pipeline,
        "hallubench": hallubench_pipeline,
        "reporting": reporting_pipeline,
    }
