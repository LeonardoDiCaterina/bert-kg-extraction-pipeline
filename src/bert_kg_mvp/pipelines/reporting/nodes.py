import logging
import random
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from bert_kg_mvp.models import build_decoder
from bert_kg_mvp.pipelines.benchmark.nodes import evaluate_model_on_test
from bert_kg_mvp.pipelines.data_prep.nodes import prepare_training_data

logger = logging.getLogger(__name__)


def generate_evaluation_report(
    split_data: Dict[str, pd.DataFrame],
    reporting_params: Dict[str, Any],
    data_prep_params: Dict[str, Any],
    schema_params: Dict[str, Any],
) -> Tuple[str, plt.Figure]:
    """
    Loads a cached model checkpoint, runs evaluation on the test set,
    and generates a Markdown report and a matplotlib Knowledge Graph visualization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Initialization
    checkpoint_path = reporting_params.get("checkpoint_path", "data/06_models/checkpoints/baseline/best_model.pt")
    encoder_name = reporting_params.get("encoder_name", "nlpaueb/sec-bert-base")
    num_examples = reporting_params.get("num_examples_to_print", 10)
    visualize_chunk_index = reporting_params.get("visualize_chunk_index", 0)

    # 2. Extract Test Data
    test_df = split_data.get("test")
    if test_df is None or test_df.empty:
        raise ValueError("Test dataset is empty or missing.")

    # 3. Load Tokenizer & Prepare Data
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    max_seq_len = data_prep_params.get("max_seq_length", 128)

    tensors, metrics = prepare_training_data(
        teacher_data=test_df,
        data_prep_params={"max_seq_length": max_seq_len},
        training_params={"encoder_model_name": encoder_name},
        schema_params=schema_params,
    )
    
    # 4. Load Model
    logger.info(f"Looking for checkpoint at: {checkpoint_path}")
    
    model = build_decoder(
        decoder_type="baseline",
        encoder_model_name=encoder_name,
        num_relations=len(schema_params.get("relation_types", ["has_metric", "produces", "operates_in", "reports_risk", "led_by"])),
        num_ent_types=len(schema_params.get("entity_types", ["org", "person", "product", "segment", "fin_metric", "risk_factor", "event"])),
        num_queries=15, # Hardcoded to baseline config for now
        num_layers=4,
        d_model=768,
        freeze_strategy="partial",
        unfrozen_top_layers=4,
    )
    
    import os
    import pickle
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as e:
            logger.info(f"torch.load failed ({e}). Falling back to standard pickle.load...")
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # Checkpoint from our ModelCheckpointer
            state_dict = checkpoint["model_state_dict"]
            unwrapped_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "").replace("module.", "")
                unwrapped_state_dict[new_key] = v
            model.load_state_dict(unwrapped_state_dict)
            logger.info("Checkpoint state_dict loaded successfully.")
        elif isinstance(checkpoint, torch.nn.Module):
            # Full model object from PickleDataset
            model = checkpoint
            logger.info("Full model object loaded successfully.")
        elif isinstance(checkpoint, dict):
            # Raw state dict
            model.load_state_dict(checkpoint)
            logger.info("Raw state_dict loaded successfully.")
            
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}. Proceeding with UNTRAINED model for smoke testing.")
        
    model.to(device)
    model.eval()

    # 5. Run Evaluation to get KPIs
    rel_to_id = {rel.lower(): i for i, rel in enumerate(schema_params.get("relation_types", ["has_metric", "produces", "operates_in", "reports_risk", "led_by"]))}
    no_relation_idx = len(rel_to_id)
    id_to_rel = {v: k for k, v in rel_to_id.items()}
    id_to_ent = {i: ent.lower() for i, ent in enumerate(schema_params.get("entity_types", ["org", "person", "product", "segment", "fin_metric", "risk_factor", "event"]))}
    
    logger.info("Running evaluation on test set...")
    val_metrics = evaluate_model_on_test(
        model=model,
        test_dataset=tensors,
        tokenizer=tokenizer,
        no_relation_idx=no_relation_idx,
        device=device,
        batch_size=16,
    )

    # 6. Generate Markdown Report
    report_lines = []
    report_lines.append(f"# Model Evaluation Report")
    report_lines.append(f"**Encoder:** `{encoder_name}`")
    report_lines.append(f"**Checkpoint:** `{checkpoint_path}`")
    report_lines.append(f"**Test Set Size:** {len(tensors['input_ids'])} chunks")
    report_lines.append("\n## Key Performance Indicators (KPIs)\n")
    report_lines.append("| Metric | Value |")
    report_lines.append("|---|---|")
    report_lines.append(f"| **Strict F1 (Exact Match)** | {val_metrics.get('strict_f1', 0.0):.4f} |")
    report_lines.append(f"| **Span F1 (Localization Only)** | {val_metrics.get('span_f1', 0.0):.4f} |")
    report_lines.append(f"| **Type Accuracy** | {val_metrics.get('type_accuracy', 0.0):.2f} |")
    report_lines.append(f"| **Relation Accuracy** | {val_metrics.get('rel_accuracy', 0.0):.2f} |")
    report_lines.append(f"| **Mean Error Distance (MED)** | {val_metrics.get('mean_error_distance', 0.0):.2f} |")
    report_lines.append(f"| **Jaccard Similarity (Mean)** | {val_metrics.get('jaccard_mean', 0.0):.2f} |")
    report_lines.append(f"| **Jaccard Similarity (Median)** | {val_metrics.get('jaccard_median', 0.0):.2f} |")
    
    report_lines.append("\n## Qualitative Examples\n")
    report_lines.append("Here is a random sample of predicted vs ground-truth tuples from the test set:")
    
    # 7. Sample Predictions
    num_test_samples = len(tensors['input_ids'])
    sample_indices = random.sample(range(num_test_samples), min(num_examples, num_test_samples))
    
    for idx in sample_indices:
        b_ids = tensors["input_ids"][idx].unsqueeze(0).to(device)
        b_mask = tensors["attention_mask"][idx].unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(b_ids, b_mask)
            
        chunk_text = tokenizer.decode(b_ids[0], skip_special_tokens=True).strip()
        report_lines.append(f"\n### Example Chunk\n> {chunk_text}\n")
        report_lines.append("| Source | Subject | Subj Type | Relation | Object | Obj Type |")
        report_lines.append("|---|---|---|---|---|---|")
        
        # Ground Truth
        for r_idx in range(len(tensors["relations"][idx])):
            if tensors["relations"][idx][r_idx] != no_relation_idx:
                s_s, s_e = tensors["subj_spans"][idx][r_idx][0], tensors["subj_spans"][idx][r_idx][1]
                o_s, o_e = tensors["obj_spans"][idx][r_idx][0], tensors["obj_spans"][idx][r_idx][1]
                subj_str = tokenizer.decode(tensors["input_ids"][idx][s_s:s_e+1], skip_special_tokens=True)
                obj_str = tokenizer.decode(tensors["input_ids"][idx][o_s:o_e+1], skip_special_tokens=True)
                r_str = id_to_rel.get(tensors["relations"][idx][r_idx].item(), str(tensors["relations"][idx][r_idx].item()))
                st_str = id_to_ent.get(tensors["subj_types"][idx][r_idx].item(), str(tensors["subj_types"][idx][r_idx].item()))
                ot_str = id_to_ent.get(tensors["obj_types"][idx][r_idx].item(), str(tensors["obj_types"][idx][r_idx].item()))
                report_lines.append(f"| ✅ GT | {subj_str} | {st_str} | {r_str} | {obj_str} | {ot_str} |")
                
        # Predictions
        rel_logits = outputs["rel_logits"][0]
        rel_preds = rel_logits.argmax(dim=-1)
        for q in range(rel_logits.shape[0]):
            pred_r = rel_preds[q].item()
            if pred_r != no_relation_idx:
                s_s = min(outputs["subj_start_logits"][0, q].argmax(), outputs["subj_end_logits"][0, q].argmax())
                s_e = max(outputs["subj_start_logits"][0, q].argmax(), outputs["subj_end_logits"][0, q].argmax())
                o_s = min(outputs["obj_start_logits"][0, q].argmax(), outputs["obj_end_logits"][0, q].argmax())
                o_e = max(outputs["obj_start_logits"][0, q].argmax(), outputs["obj_end_logits"][0, q].argmax())
                
                pred_subj = tokenizer.decode(b_ids[0][s_s:s_e+1], skip_special_tokens=True)
                pred_obj = tokenizer.decode(b_ids[0][o_s:o_e+1], skip_special_tokens=True)
                
                pred_st = outputs["subj_type_logits"][0, q].argmax().item()
                pred_ot = outputs["obj_type_logits"][0, q].argmax().item()
                
                r_str = id_to_rel.get(pred_r, str(pred_r))
                st_str = id_to_ent.get(pred_st, str(pred_st))
                ot_str = id_to_ent.get(pred_ot, str(pred_ot))
                
                report_lines.append(f"| 🤖 Pred | {pred_subj} | {st_str} | {r_str} | {pred_obj} | {ot_str} |")

    markdown_report = "\n".join(report_lines)
    
    # 8. Graph Visualization
    viz_idx = min(visualize_chunk_index, len(tensors["input_ids"]) - 1)
    b_ids = tensors["input_ids"][viz_idx].unsqueeze(0).to(device)
    b_mask = tensors["attention_mask"][viz_idx].unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(b_ids, b_mask)
        
    rel_preds = outputs["rel_logits"][0].argmax(dim=-1)
    extracted_triplets = set()
    
    for q in range(outputs["rel_logits"].shape[1]):
        r = rel_preds[q].item()
        if r != no_relation_idx:
            s_s = min(outputs["subj_start_logits"][0, q].argmax(), outputs["subj_end_logits"][0, q].argmax())
            s_e = max(outputs["subj_start_logits"][0, q].argmax(), outputs["subj_end_logits"][0, q].argmax())
            o_s = min(outputs["obj_start_logits"][0, q].argmax(), outputs["obj_end_logits"][0, q].argmax())
            o_e = max(outputs["obj_start_logits"][0, q].argmax(), outputs["obj_end_logits"][0, q].argmax())
            
            sub = tokenizer.decode(b_ids[0][s_s:s_e+1], skip_special_tokens=True)
            obj = tokenizer.decode(b_ids[0][o_s:o_e+1], skip_special_tokens=True)
            rel_str = id_to_rel.get(r, str(r))
            
            if sub and obj:
                extracted_triplets.add((sub, rel_str, obj))
                
    G = nx.DiGraph()
    for sub, rel_str, obj in extracted_triplets:
        G.add_edge(sub, obj, label=rel_str)
        
    fig = plt.figure(figsize=(10, 6))
    if not extracted_triplets:
        plt.text(0.5, 0.5, "No triples extracted for this chunk.", ha="center")
    else:
        pos = nx.spring_layout(G, k=1.0)
        nx.draw_networkx_nodes(G, pos, node_color="#87CEFA", node_size=3000, alpha=0.9)
        nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="gray", width=2)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color="red")
        plt.title(f"Extracted Knowledge Graph (Test Chunk {viz_idx})", pad=20)
        plt.axis("off")
        plt.tight_layout()

    return markdown_report, fig
