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
    if str(device) == "mps": torch.mps.empty_cache()
    gc.collect()

    trained_model.to(device)
    trained_model.eval()

    sample_size = 20
    input_ids_full = processed_dataset["input_ids"][-sample_size:]
    attention_mask_full = processed_dataset["attention_mask"][-sample_size:]
    target_ids_full = processed_dataset["labels"][-sample_size:].clone()

    bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")
    
    pred_texts = []
    batch_size = 4 
    print(f"Generating predictions for {sample_size} validation samples (Batch size: {batch_size})...")
    
    with torch.no_grad():
        for i in range(0, sample_size, batch_size):
            input_ids = input_ids_full[i:i+batch_size].to(device)
            attention_mask = attention_mask_full[i:i+batch_size].to(device)
            
            current_bs = input_ids.size(0)
            decoder_input_ids = torch.full((current_bs, 1), bos_token_id, dtype=torch.long, device=device)
            
            for _ in range(60):
                logits = trained_model(input_ids, attention_mask, decoder_input_ids)
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)
                
                if (decoder_input_ids == eos_token_id).any(dim=1).all():
                    break
                    
            pred_texts.extend(tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=False))
            del input_ids, attention_mask, decoder_input_ids, logits
            if str(device) == "mps": torch.mps.empty_cache()

    target_ids_full[target_ids_full == -100] = tokenizer.pad_token_id
    true_texts = tokenizer.batch_decode(target_ids_full, skip_special_tokens=False)

    true_positives, false_positives, false_negatives = 0, 0, 0

    for idx, (pred, true) in enumerate(zip(pred_texts, true_texts)):
        pred_set = parse_triplet_string(pred)
        true_set = parse_triplet_string(true)

        true_positives += len(pred_set & true_set)
        false_positives += len(pred_set - true_set)
        false_negatives += len(true_set - pred_set)
        
        if idx < 3:
            print(f"\n--- Sample {idx+1} ---")
            print(f"TARGET:    {true_set}")
            print(f"PREDICTED: {pred_set}")

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
