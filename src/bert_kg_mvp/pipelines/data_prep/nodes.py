import json
import os
import re
import torch
import pandas as pd
from transformers import BertTokenizer
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

def prepare_training_data(teacher_data: pd.DataFrame, parameters: dict):
    max_samples = parameters.get("max_samples", 5000)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    special_tokens = ['[BOS]', '[EOS]', '<triplet>', '<subj_type>', '<relation>', '<obj>', '<obj_type>']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    input_texts, target_texts = [], []
    
    for _, row in teacher_data.iterrows():
        # Handle parsed JSON lists or raw string representations
        try:
            triplets = json.loads(row["triples"].replace("'", '"')) if isinstance(row["triples"], str) else row["triples"]
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
            
        if not triplets:
            continue
            
        tgt = "[BOS] "
        valid = False
        for t in triplets:
            sub = t.get("head", "").strip().lower()

            if sub == "exact company name":
                if "AAPL" in row.get("doc_id", ""):
                    sub = "apple inc."
                elif "MSFT" in row.get("doc_id", ""):
                    sub = "microsoft corp."
                else:
                    sub = "company"
            sub_type = t.get("head_type", "entity").strip().lower()
            rel = t.get("relation", "").strip().lower()
            obj = t.get("tail", "").strip().lower()
            obj_type = t.get("tail_type", "entity").strip().lower()
            
            if sub and rel and obj:
                tgt += f"<triplet> {sub} <subj_type> {sub_type} <relation> {rel} <obj> {obj} <obj_type> {obj_type} "
                valid = True
                
        if not valid:
            continue
            
        tgt += "[EOS]"
        input_texts.append(row["text"])
        target_texts.append(tgt)
        
        if len(input_texts) >= max_samples:
            break
            
    print(f"Gathered {len(input_texts)} valid financial samples.")
            
    encodings_input = tokenizer(input_texts, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    encodings_target = tokenizer(target_texts, add_special_tokens=False, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    
    decoder_input_ids = encodings_target['input_ids'][:, :-1]
    labels = encodings_target['input_ids'][:, 1:].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": encodings_input['input_ids'],
        "attention_mask": encodings_input['attention_mask'],
        "decoder_input_ids": decoder_input_ids,
        "labels": labels
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