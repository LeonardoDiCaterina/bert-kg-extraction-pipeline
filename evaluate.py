import re
import torch
from pathlib import Path
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession

def parse_triplet_string(text):
    """Uses regex to robustly parse the (subject, relation, object) tuples."""
    triplets = set()
    # Safely matches everything between the specific structural tags
    pattern = r"<triplet>\s*(.*?)\s*<subj_type>.*?<relation>\s*(.*?)\s*<obj>\s*(.*?)\s*<obj_type>"
    matches = re.findall(pattern, text)
    
    for sub, rel, obj in matches:
        if sub and rel and obj:
            triplets.add((sub.strip(), rel.strip(), obj.strip()))
            
    return triplets

def evaluate_baseline():
    bootstrap_project(Path.cwd())
    with KedroSession.create() as session:
        context = session.load_context()
        tokenizer = context.catalog.load("kg_tokenizer")
        model = context.catalog.load("trained_model")
        dataset = context.catalog.load("processed_dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device)
    model.eval()

    sample_size = 20
    # Evaluate on the LAST 20 samples to ensure they weren't heavily memorized in the first batches
    input_ids = dataset["input_ids"][-sample_size:].to(device)
    attention_mask = dataset["attention_mask"][-sample_size:].to(device)
    target_ids = dataset["labels"][-sample_size:].clone()

    bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")
    decoder_input_ids = torch.full((sample_size, 1), bos_token_id, dtype=torch.long, device=device)

    print(f"Generating predictions for {sample_size} validation samples...")
    with torch.no_grad():
        for _ in range(60):
            logits = model(input_ids, attention_mask, decoder_input_ids)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)
            
            if (decoder_input_ids == eos_token_id).any(dim=1).all():
                break

    pred_texts = tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True)

    target_ids[target_ids == -100] = tokenizer.pad_token_id
    true_texts = tokenizer.batch_decode(target_ids, skip_special_tokens=True)

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

if __name__ == "__main__":
    evaluate_baseline()
