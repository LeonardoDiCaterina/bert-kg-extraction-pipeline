import json
import os
import torch
import pandas as pd
from transformers import BertTokenizer
from docling.document_converter import DocumentConverter

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
    """Parses SEC 10-K PDFs into table-aware markdown chunks."""
    converter = DocumentConverter()
    all_chunks = []
    
    # Iterate through all PDFs in the raw data directory
    for filename in os.listdir(raw_data_dir):
        if not filename.endswith(".pdf"):
            continue
            
        pdf_path = os.path.join(raw_data_dir, filename)
        print(f"Extracting {filename}...")
        doc = converter.convert(pdf_path).document
        
        current_chunk = ""
        chunk_idx = 0
        
        for item, level in doc.iterate_items():
            if item.label == "table":
                if current_chunk.strip():
                    all_chunks.append({"doc_id": filename, "chunk_id": chunk_idx, "text": current_chunk.strip()})
                    chunk_idx += 1
                    current_chunk = ""
                
                # Pass 'doc' to silence the deprecation warning
                table_md = item.export_to_markdown(doc)
                table_text = f"[TABLE START]\n{table_md}\n[TABLE END]"
                all_chunks.append({"doc_id": filename, "chunk_id": chunk_idx, "text": table_text})
                chunk_idx += 1
                
            elif hasattr(item, 'text') and item.text:
                current_chunk += f"{item.text}\n"
                
                if len(current_chunk.split()) > max_words:
                    all_chunks.append({"doc_id": filename, "chunk_id": chunk_idx, "text": current_chunk.strip()})
                    chunk_idx += 1
                    current_chunk = ""
                    
        if current_chunk.strip():
            all_chunks.append({"doc_id": filename, "chunk_id": chunk_idx, "text": current_chunk.strip()})
            
    return pd.DataFrame(all_chunks)