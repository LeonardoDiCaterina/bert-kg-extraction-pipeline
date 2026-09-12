import gc
from typing import Any, Dict, Optional
import torch
import pandas as pd

from bert_kg_mvp.utils import parse_triplet_string

# Backward-compatible re-export
__all__ = ["parse_triplet_string", "run_mvp_inference"]


def run_mvp_inference(
    processed_dataset: Dict[str, torch.Tensor],
    tokenizer: Any,
    trained_model: Any,
    inference_params: Dict[str, Any],
    schema_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Runs set-prediction inference and evaluates Precision, Recall, and F1 score against ground truth.
    Supports either namespaced params (`params:inference` + `params:schema`) or a legacy single dict.
    """
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    if str(device) == "mps":
        torch.mps.empty_cache()
    gc.collect()

    trained_model.to(device)
    trained_model.eval()

    # Dynamic schema handling
    if schema_params is None:
        schema_params = inference_params.get("schema", {})
    relation_types = schema_params.get(
        "relation_types",
        ["has_metric", "produces", "operates_in", "reports_risk", "led_by"],
    )
    no_relation_idx = len(relation_types)

    # Dynamic parameterization
    sample_size_param = inference_params.get("sample_size", 20)
    batch_size = inference_params.get("batch_size", 4)

    total_samples = len(processed_dataset["input_ids"])
    sample_size = min(sample_size_param, total_samples)

    input_ids_full = processed_dataset["input_ids"][-sample_size:]
    attention_mask_full = processed_dataset["attention_mask"][-sample_size:]

    gt_rels = processed_dataset["relations"][-sample_size:]
    gt_subj_spans = processed_dataset["subj_spans"][-sample_size:]
    gt_obj_spans = processed_dataset["obj_spans"][-sample_size:]

    print(
        f"Generating predictions for {sample_size} validation samples (Batch size: {batch_size})..."
    )

    true_positives, false_positives, false_negatives = 0, 0, 0

    with torch.no_grad():
        for i in range(0, sample_size, batch_size):
            input_ids = input_ids_full[i : i + batch_size].to(device)
            attention_mask = attention_mask_full[i : i + batch_size].to(device)

            if input_ids.size(0) == 0:
                continue

            outputs = trained_model(input_ids, attention_mask)

            rel_preds = torch.argmax(outputs["rel_logits"], dim=-1)
            subj_start_preds = torch.argmax(outputs["subj_start_logits"], dim=-1)
            subj_end_preds = torch.argmax(outputs["subj_end_logits"], dim=-1)
            obj_start_preds = torch.argmax(outputs["obj_start_logits"], dim=-1)
            obj_end_preds = torch.argmax(outputs["obj_end_logits"], dim=-1)

            for b in range(input_ids.size(0)):
                pred_set = set()
                true_set = set()

                valid_gt = gt_rels[i + b] != no_relation_idx
                for r, ss, os in zip(
                    gt_rels[i + b][valid_gt],
                    gt_subj_spans[i + b][valid_gt],
                    gt_obj_spans[i + b][valid_gt],
                ):
                    subj_str = tokenizer.decode(
                        input_ids[b, ss[0] : ss[1] + 1], skip_special_tokens=True
                    ).strip()
                    obj_str = tokenizer.decode(
                        input_ids[b, os[0] : os[1] + 1], skip_special_tokens=True
                    ).strip()
                    true_set.add((subj_str, r.item(), obj_str))

                num_queries = outputs["rel_logits"].shape[1]
                for q in range(num_queries):
                    rel = rel_preds[b, q].item()
                    if rel != no_relation_idx:
                        s_start = min(
                            subj_start_preds[b, q].item(), subj_end_preds[b, q].item()
                        )
                        s_end = max(
                            subj_start_preds[b, q].item(), subj_end_preds[b, q].item()
                        )
                        o_start = min(
                            obj_start_preds[b, q].item(), obj_end_preds[b, q].item()
                        )
                        o_end = max(
                            obj_start_preds[b, q].item(), obj_end_preds[b, q].item()
                        )

                        pred_subj = tokenizer.decode(
                            input_ids[b, s_start : s_end + 1], skip_special_tokens=True
                        ).strip()
                        pred_obj = tokenizer.decode(
                            input_ids[b, o_start : o_end + 1], skip_special_tokens=True
                        ).strip()

                        if pred_subj and pred_obj:
                            pred_set.add((pred_subj, rel, pred_obj))

                tp = len(pred_set.intersection(true_set))
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)

                true_positives += tp
                false_positives += fp
                false_negatives += fn

                if (i + b) < 3:
                    print(f"\n--- Sample {i + b + 1} ---")
                    print(f"TARGET:    {true_set}")
                    print(f"PREDICTED: {pred_set}")

            del input_ids, attention_mask, outputs
            if str(device) == "mps":
                torch.mps.empty_cache()

    precision = true_positives / max((true_positives + false_positives), 1)
    recall = true_positives / max((true_positives + false_negatives), 1)
    f1 = 2 * (precision * recall) / max((precision + recall), 1e-9)

    print("\n" + "=" * 40)
    print(f"Validation F1 Score: {f1:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print("=" * 40 + "\n")

    return pd.DataFrame([{"f1_score": f1, "precision": precision, "recall": recall}])
