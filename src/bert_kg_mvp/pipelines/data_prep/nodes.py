import json
import torch
import pandas as pd
from transformers import AutoTokenizer

def serialize_and_sort_triplets(row):
    # Parse JSON string to list of dicts
    triplets = json.loads(row['triplets']) if isinstance(row['triplets'], str) else row['triplets']
    
    # 1. Canonical Ordering: Sort alphabetically by head, relation, tail
    sorted_triplets = sorted(triplets, key=lambda x: (x['head'], x['relation'], x['tail']))
    
    # 2. Syntax Formatting
    triplet_strs = []
    for t in sorted_triplets:
        t_str = f"{t['head']} | {t['head_type']} | {t['relation']} | {t['tail']} | {t['tail_type']} [EOT]"
        triplet_strs.append(t_str)
    
    # Return string ending with EOS
    return " ".join(triplet_strs) + " [EOS]"

def prepare_training_data(raw_df: pd.DataFrame, parameters: dict):
    # Apply serialization
    raw_df['target_string'] = raw_df.apply(serialize_and_sort_triplets, axis=1)
    
    # Initialize tokenizer and expand vocabulary
    tokenizer = AutoTokenizer.from_pretrained(parameters["model_name"])
    special_tokens = {'additional_special_tokens': ['[BOS]', '[EOS]', '[EOT]', '|']}
    tokenizer.add_special_tokens(special_tokens)
    
    # Tokenize Encoder Inputs (Context)
    encoder_encodings = tokenizer(
        raw_df["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=parameters["max_seq_len"],
        return_tensors="pt"
    )
    
    # Tokenize Decoder Inputs (Prepending [BOS] to target strings)
    bos_targets = ["[BOS] " + tgt for tgt in raw_df['target_string']]
    decoder_encodings = tokenizer(
        bos_targets,
        padding="max_length",
        truncation=True,
        max_length=parameters["max_seq_len"],
        return_tensors="pt"
    )
    
    # Create Teacher Forcing Alignment
    # decoder_input_ids: Tokens t_0 to t_{N-1}
    # labels: Tokens t_1 to t_N (Shifted right by 1)
    decoder_input_ids = decoder_encodings["input_ids"][:, :-1]
    labels = decoder_encodings["input_ids"][:, 1:].clone()
    
    # Mask padding tokens with -100 so CrossEntropyLoss ignores them
    pad_token_id = tokenizer.pad_token_id
    labels[labels == pad_token_id] = -100
    
    processed_data = {
        "input_ids": encoder_encodings["input_ids"],
        "attention_mask": encoder_encodings["attention_mask"],
        "decoder_input_ids": decoder_input_ids,
        "labels": labels
    }
    
    return processed_data, tokenizer
