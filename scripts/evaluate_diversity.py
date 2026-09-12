import json
import re
import glob
import math
import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import entropy

# Pre-defined schema sizes from parameters.yml
NUM_ENTITY_TYPES = 10
NUM_RELATION_TYPES = 13

def clean_triples_string(raw_s):
    """Applies the same regex fix as nodes.py to fix missing commas."""
    if not isinstance(raw_s, str) or len(raw_s) < 5:
        return raw_s
    # Fix single quotes to double quotes, handle boolean/None stringification
    raw_s = raw_s.replace("'", '"')
    raw_s = raw_s.replace("True", "true").replace("False", "false").replace("None", "null")
    # Repair missing commas between dictionary objects
    raw_s = re.sub(r"\}\s*\{", "}, {", raw_s)
    return raw_s

def compute_renyi_entropy(probs, alpha=2):
    """Computes Rényi Entropy for alpha=2 (Collision Entropy)."""
    if len(probs) == 0:
        return 0.0
    sum_sq = np.sum(probs ** alpha)
    if sum_sq == 0:
        return 0.0
    return -math.log2(sum_sq)

def main():
    print("Loading teacher batches...")
    batch_files = glob.glob("data/02_intermediate/teacher_batches/*.parquet")
    
    if not batch_files:
        print("No batch files found in data/02_intermediate/teacher_batches/")
        return
        
    df_list = [pd.read_parquet(f) for f in batch_files]
    df = pd.concat(df_list, ignore_index=True)
    
    total_chunks = len(df)
    print(f"Loaded {total_chunks} chunks.")
    
    all_entities = []
    all_entity_types = []
    all_relations = []
    total_valid_triples = 0
    non_empty_chunks = 0
    
    for _, row in df.iterrows():
        raw = clean_triples_string(row.get("triples", "[]"))
        try:
            triples = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(triples, list):
                continue
                
            if len(triples) > 0:
                non_empty_chunks += 1
                
            for t in triples:
                if not isinstance(t, dict): continue
                
                if "head" in t: all_entities.append(str(t["head"]).lower().strip())
                if "tail" in t: all_entities.append(str(t["tail"]).lower().strip())
                if "head_type" in t: all_entity_types.append(str(t["head_type"]))
                if "tail_type" in t: all_entity_types.append(str(t["tail_type"]))
                if "relation" in t: all_relations.append(str(t["relation"]))
                total_valid_triples += 1
                
        except Exception:
            continue
            
    # Calculate Ratios
    triples_per_chunk = total_valid_triples / max(1, total_chunks)
    unique_entities = len(set(all_entities))
    unique_entity_types = len(set(all_entity_types))
    unique_relations = len(set(all_relations))
    
    ecr = unique_entities / max(1, len(all_entities))
    tcr_n = unique_entity_types / NUM_ENTITY_TYPES
    rcr_n = unique_relations / NUM_RELATION_TYPES
    
    # Calculate Frequencies and Entropy
    ent_type_counts = list(Counter(all_entity_types).values())
    rel_counts = list(Counter(all_relations).values())
    
    ent_type_probs = np.array(ent_type_counts) / max(1, sum(ent_type_counts))
    rel_probs = np.array(rel_counts) / max(1, sum(rel_counts))
    
    ent_shannon = entropy(ent_type_counts, base=2) if ent_type_counts else 0
    rel_shannon = entropy(rel_counts, base=2) if rel_counts else 0
    
    ent_norm_shannon = ent_shannon / math.log2(NUM_ENTITY_TYPES) if NUM_ENTITY_TYPES > 1 else 0
    rel_norm_shannon = rel_shannon / math.log2(NUM_RELATION_TYPES) if NUM_RELATION_TYPES > 1 else 0
    
    ent_renyi = compute_renyi_entropy(ent_type_probs)
    rel_renyi = compute_renyi_entropy(rel_probs)
    
    print("\n" + "="*50)
    print(f"Table 5: Local Extraction Efficiency (N={total_chunks} chunks)")
    print("="*50)
    print(f"Triples (per chunk) : {triples_per_chunk:.2f}")
    print(f"ECR                 : {ecr:.4f} (Unique: {unique_entities} / Total: {len(all_entities)})")
    print(f"TCR-N               : {tcr_n:.4f} (Found: {unique_entity_types} / Schema: {NUM_ENTITY_TYPES})")
    print(f"RCR-N               : {rcr_n:.4f} (Found: {unique_relations} / Schema: {NUM_RELATION_TYPES})")
    
    print("\n" + "="*50)
    print("Table 6: Global Semantic Diversity (Entropy)")
    print("="*50)
    print("--- Shannon Entropy ---")
    print(f"Entity Type  : {ent_shannon:.4f}")
    print(f"Relationship : {rel_shannon:.4f}")
    
    print("\n--- Normalized Schema Entropy ---")
    print(f"Entity Type  : {ent_norm_shannon:.4f}")
    print(f"Relationship : {rel_norm_shannon:.4f}")
    
    print("\n--- Rényi Entropy (alpha=2) ---")
    print(f"Entity Type  : {ent_renyi:.4f}")
    print(f"Relationship : {rel_renyi:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
