import sys
from pathlib import Path
from typing import Set, Tuple
import torch
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession

from visualize_graph import parse_and_visualize


def predict_triplets(text: str) -> Set[Tuple[str, str, str]]:
    """Predicts KG triplets for an input text using the trained DynamicKGExtractor model."""
    bootstrap_project(Path.cwd())
    with KedroSession.create() as session:
        context = session.load_context()
        tokenizer = context.catalog.load("kg_tokenizer")
        model = context.catalog.load("trained_model")
        params = context.params

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device)
    model.eval()

    schema = params.get("schema", {})
    relation_types = schema.get("relation_types", ["has_metric", "produces", "operates_in", "reports_risk", "led_by"])
    no_relation_idx = len(relation_types)

    max_len = params.get("training", {}).get("max_seq_len", 128)
    encodings = tokenizer([text], padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    triplets: Set[Tuple[str, str, str]] = set()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        rel_preds = torch.argmax(outputs["rel_logits"], dim=-1)[0]
        subj_start = torch.argmax(outputs["subj_start_logits"], dim=-1)[0]
        subj_end = torch.argmax(outputs["subj_end_logits"], dim=-1)[0]
        obj_start = torch.argmax(outputs["obj_start_logits"], dim=-1)[0]
        obj_end = torch.argmax(outputs["obj_end_logits"], dim=-1)[0]

        num_queries = outputs["rel_logits"].shape[1]
        for q in range(num_queries):
            rel = rel_preds[q].item()
            if rel != no_relation_idx and rel < len(relation_types):
                rel_name = relation_types[rel]
                s_s = min(subj_start[q].item(), subj_end[q].item())
                s_e = max(subj_start[q].item(), subj_end[q].item())
                o_s = min(obj_start[q].item(), obj_end[q].item())
                o_e = max(obj_start[q].item(), obj_end[q].item())

                subj_str = tokenizer.decode(input_ids[0, s_s:s_e + 1], skip_special_tokens=True).strip()
                obj_str = tokenizer.decode(input_ids[0, o_s:o_e + 1], skip_special_tokens=True).strip()
                if subj_str and obj_str:
                    triplets.add((subj_str, rel_name, obj_str))

    print(f"\n[INPUT]:  {text}")
    print(f"[PREDICTED TRIPLETS]: {triplets}\n")

    print("Drawing Knowledge Graph...")
    parse_and_visualize(triplets)

    return triplets


if __name__ == "__main__":
    test_text = sys.argv[1] if len(sys.argv) > 1 else "Elon Musk founded SpaceX. SpaceX is located in California."
    predict_triplets(test_text)
