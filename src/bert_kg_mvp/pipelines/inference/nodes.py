import gc
import re
import torch
import pandas as pd

def parse_triplet_string(text):
    triplets = set()
    pattern = r"<triplet>\s*(.*?)\s*<subj_type>.*?<relation>\s*(.*?)\s*<obj>\s*(.*?)\s*<obj_type>"
    matches = re.findall(pattern, text)
    for sub, rel, obj in matches:
        if sub and rel and obj:
            triplets.add((sub.strip(), rel.strip(), obj.strip()))
    return triplets

def run_mvp_inference(processed_dataset: dict, tokenizer, trained_model, parameters: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    if str(device) == "mps":
        torch.mps.empty_cache()
    gc.collect()

    trained_model.to(device)
    trained_model.eval()

    sample_size = min(20, len(processed_dataset["input_ids"]))
    input_ids_full = processed_dataset["input_ids"][-sample_size:]
    attention_mask_full = processed_dataset["attention_mask"][-sample_size:]
    
    # Ground truth for evaluation
    gt_rels = processed_dataset["relations"][-sample_size:]
    gt_subj_spans = processed_dataset["subj_spans"][-sample_size:]
    gt_obj_spans = processed_dataset["obj_spans"][-sample_size:]

    batch_size = 4 
    print(f"Generating predictions for {sample_size} validation samples (Batch size: {batch_size})...")
    
    true_positives, false_positives, false_negatives = 0, 0, 0
    
    with torch.no_grad():
        for i in range(0, sample_size, batch_size):
            input_ids = input_ids_full[i:i+batch_size].to(device)
            attention_mask = attention_mask_full[i:i+batch_size].to(device)
            
            # Skip empty batches to prevent PyTorch reshape errors
            if input_ids.size(0) == 0:
                continue
                
            outputs = trained_model(input_ids, attention_mask)
            
            rel_preds = torch.argmax(outputs["rel_logits"], dim=-1) # [B, num_queries]
            subj_start_preds = torch.argmax(outputs["subj_start_logits"], dim=-1)
            subj_end_preds = torch.argmax(outputs["subj_end_logits"], dim=-1)
            obj_start_preds = torch.argmax(outputs["obj_start_logits"], dim=-1)
            obj_end_preds = torch.argmax(outputs["obj_end_logits"], dim=-1)
            
            for b in range(input_ids.size(0)):
                pred_set = set()
                true_set = set()
                
                # Extract ground truth set
                valid_gt = gt_rels[i+b] != 5 # 5 is no_relation
                for r, ss, os in zip(gt_rels[i+b][valid_gt], gt_subj_spans[i+b][valid_gt], gt_obj_spans[i+b][valid_gt]):
                    subj_str = tokenizer.decode(input_ids[b, ss[0]:ss[1]+1], skip_special_tokens=True).strip()
                    obj_str = tokenizer.decode(input_ids[b, os[0]:os[1]+1], skip_special_tokens=True).strip()
                    true_set.add((subj_str, r.item(), obj_str))
                    
                # Extract predicted set
                for q in range(trained_model.num_queries):
                    r_pred = rel_preds[b, q].item()
                    if r_pred == 5: # no_relation
                        continue
                        
                    ss = subj_start_preds[b, q].item()
                    se = subj_end_preds[b, q].item()
                    os = obj_start_preds[b, q].item()
                    oe = obj_end_preds[b, q].item()
                    
                    if se < ss or oe < os:
                        continue # Invalid spans
                        
                    subj_str = tokenizer.decode(input_ids[b, ss:se+1], skip_special_tokens=True).strip()
                    obj_str = tokenizer.decode(input_ids[b, os:oe+1], skip_special_tokens=True).strip()
                    pred_set.add((subj_str, r_pred, obj_str))
                    
                true_positives += len(pred_set & true_set)
                false_positives += len(pred_set - true_set)
                false_negatives += len(true_set - pred_set)
                
                if (i + b) < 3:
                    print(f"\n--- Sample {i+b+1} ---")
                    print(f"TARGET:    {true_set}")
                    print(f"PREDICTED: {pred_set}")
            
            del input_ids, attention_mask, outputs
            if str(device) == "mps":
                torch.mps.empty_cache()

    precision = true_positives / max((true_positives + false_positives), 1)
    recall = true_positives / max((true_positives + false_negatives), 1)
    f1 = 2 * (precision * recall) / max((precision + recall), 1e-9)

    print("\n" + "="*40)
    print(f"Validation F1 Score: {f1:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print("="*40 + "\n")
    
    # FIX: Return a Pandas DataFrame so the Kedro CSVDataset catalog hook succeeds
    return pd.DataFrame([{"f1_score": f1, "precision": precision, "recall": recall}])
