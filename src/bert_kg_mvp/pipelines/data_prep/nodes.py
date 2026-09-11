import json
import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter

# High-signal sections in SEC 10-K filings
TARGET_ITEMS_PATTERN = re.compile(
    r"(item\s+(1|1a|7|7a|8)\.?\s+)", 
    re.IGNORECASE
)
STOP_ITEMS_PATTERN = re.compile(
    r"(item\s+(9|10|15)\.?\s+|signat(ure|ures)|part\s+iv)", 
    re.IGNORECASE
)

def is_informative_chunk(text: str) -> bool:
    """Filters out empty tables, legal boilerplate, and short snippets."""
    words = text.split()
    if len(words) < 50:
        return False
    # Drop checkbox-heavy administrative blocks
    if "indicate by check mark" in text.lower():
        return False
    return True

def extract_rebel_triplets(text):
    triplets = []
    relation, subject, object_ = '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
    return triplets

def align_entities_to_tokens(text: str, entity_str: str, tokenizer, input_ids):
    """Finds the start and end token indices of entity_str in the tokenized text."""
    # Tokenize the entity without special tokens
    ent_ids = tokenizer.encode(entity_str, add_special_tokens=False)
    if not ent_ids:
        return -1, -1
        
    seq = input_ids.tolist()
    # Sliding window search for exact token match
    for i in range(len(seq) - len(ent_ids) + 1):
        if seq[i:i+len(ent_ids)] == ent_ids:
            return i, i + len(ent_ids) - 1
            
    # Fallback: if exact token sequence isn't found (due to spacing/punctuation differences)
    # This is a naive fallback - ideally we'd use character offsets
    return -1, -1

def prepare_training_data(teacher_data: pd.DataFrame, parameters: dict):
    max_samples = parameters.get("max_samples", 5000)
    max_gt_triples = parameters.get("max_gt_triples", 15)
    max_seq_length = parameters.get("max_seq_length", 128)
    encoder_model_name = parameters.get("encoder_model_name", "bert-base-uncased")
    
    tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
    
    ENTITY_TYPES = ["org", "person", "product", "segment", "fin_metric", "risk_factor", "event"]
    RELATION_TYPES = ["has_metric", "produces", "operates_in", "reports_risk", "led_by"]
    
    ent_to_id = {ent: i for i, ent in enumerate(ENTITY_TYPES)}
    rel_to_id = {rel: i for i, rel in enumerate(RELATION_TYPES)}
    
    # Store schema maps in tokenizer for downstream use
    tokenizer.ent_to_id = ent_to_id
    tokenizer.rel_to_id = rel_to_id
    
    input_ids_list, attention_masks_list = [], []
    gt_relations_list = []
    gt_subj_types_list, gt_obj_types_list = [], []
    gt_subj_spans_list, gt_obj_spans_list = [], []
    
    for _, row in teacher_data.iterrows():
        try:
            triplets = json.loads(row["triples"].replace("'", '"')) if isinstance(row["triples"], str) else row["triples"]
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
            
        if not triplets:
            continue
            
        text = row["text"]
        encodings = tokenizer(text, padding='max_length', max_length=max_seq_length, truncation=True, return_tensors="pt")
        ids = encodings['input_ids'][0]
        mask = encodings['attention_mask'][0]
        
        valid_triples = []
        for t in triplets:
            sub = t.get("head", "").strip().lower()
            if sub == "exact company name":
                sub = "apple inc." if "AAPL" in row.get("doc_id", "") else ("microsoft corp." if "MSFT" in row.get("doc_id", "") else "company")
                
            sub_type = t.get("head_type", "").strip().lower()
            rel = t.get("relation", "").strip().lower()
            obj = t.get("tail", "").strip().lower()
            obj_type = t.get("tail_type", "").strip().lower()
            
            if sub_type not in ent_to_id or obj_type not in ent_to_id or rel not in rel_to_id:
                continue
                
            subj_start, subj_end = align_entities_to_tokens(text, sub, tokenizer, ids)
            obj_start, obj_end = align_entities_to_tokens(text, obj, tokenizer, ids)
            
            # If both entities are found in the text
            if subj_start != -1 and obj_start != -1:
                valid_triples.append({
                    "relation": rel_to_id[rel],
                    "subj_type": ent_to_id[sub_type],
                    "obj_type": ent_to_id[obj_type],
                    "subj_span": [subj_start, subj_end],
                    "obj_span": [obj_start, obj_end]
                })
                
        if not valid_triples:
            continue
            
        # Pad triples up to max_gt_triples (or truncate if too many)
        valid_triples = valid_triples[:max_gt_triples]
        
        rel_tensor = torch.full((max_gt_triples,), len(RELATION_TYPES), dtype=torch.long) # default to 'no_relation' (idx=len)
        subj_type_tensor = torch.zeros((max_gt_triples,), dtype=torch.long)
        obj_type_tensor = torch.zeros((max_gt_triples,), dtype=torch.long)
        subj_span_tensor = torch.zeros((max_gt_triples, 2), dtype=torch.long)
        obj_span_tensor = torch.zeros((max_gt_triples, 2), dtype=torch.long)
        
        for i, vt in enumerate(valid_triples):
            rel_tensor[i] = vt["relation"]
            subj_type_tensor[i] = vt["subj_type"]
            obj_type_tensor[i] = vt["obj_type"]
            subj_span_tensor[i] = torch.tensor(vt["subj_span"])
            obj_span_tensor[i] = torch.tensor(vt["obj_span"])
            
        input_ids_list.append(ids)
        attention_masks_list.append(mask)
        gt_relations_list.append(rel_tensor)
        gt_subj_types_list.append(subj_type_tensor)
        gt_obj_types_list.append(obj_type_tensor)
        gt_subj_spans_list.append(subj_span_tensor)
        gt_obj_spans_list.append(obj_span_tensor)
        
        if len(input_ids_list) >= max_samples:
            break
            
    print(f"Gathered {len(input_ids_list)} valid financial samples.")
    
    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_masks_list),
        "relations": torch.stack(gt_relations_list),
        "subj_types": torch.stack(gt_subj_types_list),
        "obj_types": torch.stack(gt_obj_types_list),
        "subj_spans": torch.stack(gt_subj_spans_list),
        "obj_spans": torch.stack(gt_obj_spans_list)
    }, tokenizer

def parse_sec_filings(raw_data_dir: str, max_words: int = 1500) -> pd.DataFrame:
    """Parses SEC 10-K PDFs into table-aware markdown chunks, filtering out boilerplate."""
    converter = DocumentConverter()
    all_chunks = []
    
    for filename in os.listdir(raw_data_dir):
        if not filename.endswith(".pdf"):
            continue
            
        pdf_path = os.path.join(raw_data_dir, filename)
        print(f"Extracting {filename}...")
        try:
            doc = converter.convert(pdf_path).document
        except Exception as e:
            print(f"Skipping {filename} due to parse error: {e}")
            continue
        
        current_chunk = ""
        chunk_idx = 0
        in_target_section = False
        
        for item, level in doc.iterate_items():
            # Check for section toggles
            if hasattr(item, 'text') and item.text:
                if TARGET_ITEMS_PATTERN.search(item.text):
                    in_target_section = True
                elif STOP_ITEMS_PATTERN.search(item.text):
                    in_target_section = False
            
            # Skip if we are in administrative boilerplate
            if not in_target_section:
                continue
                
            if item.label == "table":
                if current_chunk.strip():
                    if is_informative_chunk(current_chunk):
                        all_chunks.append({"doc_id": filename, "chunk_id": chunk_idx, "text": current_chunk.strip()})
                        chunk_idx += 1
                    current_chunk = ""
                
                table_md = item.export_to_markdown(doc)
                table_text = f"[TABLE START]\n{table_md}\n[TABLE END]"
                all_chunks.append({"doc_id": filename, "chunk_id": chunk_idx, "text": table_text})
                chunk_idx += 1
                
            elif hasattr(item, 'text') and item.text:
                current_chunk += f"{item.text}\n"
                
                if len(current_chunk.split()) > max_words:
                    if is_informative_chunk(current_chunk):
                        all_chunks.append({"doc_id": filename, "chunk_id": chunk_idx, "text": current_chunk.strip()})
                        chunk_idx += 1
                    current_chunk = ""
                    
        if current_chunk.strip() and is_informative_chunk(current_chunk):
            all_chunks.append({"doc_id": filename, "chunk_id": chunk_idx, "text": current_chunk.strip()})
            
    return pd.DataFrame(all_chunks)