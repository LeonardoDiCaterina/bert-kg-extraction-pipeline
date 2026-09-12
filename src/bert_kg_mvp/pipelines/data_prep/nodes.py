import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Optional, Tuple, Union
import torch
import pandas as pd
from transformers import AutoTokenizer

from bert_kg_mvp.utils import (
    align_entities_to_tokens,
    extract_html_from_sgml,
    extract_rebel_triplets,
    infer_section_label,
    is_informative_chunk,
    parse_doc_metadata,
)

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None  # type: ignore[assignment, misc]

# Backward-compatible re-exports
__all__ = [
    "is_informative_chunk",
    "extract_rebel_triplets",
    "align_entities_to_tokens",
    "prepare_training_data",
    "parse_sec_filings",
]


def _ensure_provenance_metadata(
    df: pd.DataFrame, data_prep_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Self-healing provenance enrichment:
    If teacher triplets were generated without doc_id/ticker/year/section metadata
    (e.g., from an earlier pipeline run), this function automatically recovers provenance
    by aligning with the intermediate parsed_10k_chunks dataset or running regex parsers.
    """
    df = df.copy()

    # 1. Recover doc_id if missing or completely empty
    needs_doc_id = (
        "doc_id" not in df.columns or not df["doc_id"].astype(str).str.strip().any()
    )
    if needs_doc_id:
        custom_chunks_path = data_prep_params.get("parsed_chunks_path")
        candidate_paths = [Path(custom_chunks_path)] if custom_chunks_path else []
        candidate_paths.extend(
            [
                Path("data/02_intermediate/parsed_10k_chunks.csv"),
                Path("data/02_intermediate/parsed_10k_chunks.parquet"),
            ]
        )
        for p in candidate_paths:
            if p.exists():
                try:
                    chunks_df = (
                        pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                    )
                    if "doc_id" in chunks_df.columns:
                        if len(chunks_df) == len(df):
                            df["doc_id"] = chunks_df["doc_id"].values
                            print(
                                f"[Auto-Enrich] Reconciled doc_id from {p} via 1:1 row alignment."
                            )
                            break
                        elif (
                            "chunk_id" in df.columns and "chunk_id" in chunks_df.columns
                        ):
                            mapping = chunks_df.drop_duplicates("chunk_id").set_index(
                                "chunk_id"
                            )["doc_id"]
                            df["doc_id"] = df["chunk_id"].map(mapping).fillna("")
                            print(
                                f"[Auto-Enrich] Reconciled doc_id from {p} via chunk_id mapping."
                            )
                            break
                except Exception as exc:
                    print(f"Notice: Could not load candidate chunks from {p}: {exc}")

    # 2. Extract ticker and year from doc_id
    has_ticker = "ticker" in df.columns and df["ticker"].astype(str).str.strip().any()
    if not has_ticker and "doc_id" in df.columns:
        parsed = df["doc_id"].astype(str).apply(parse_doc_metadata)
        df["ticker"] = [m.get("ticker", "") for m in parsed]
        if "year" not in df.columns or not df["year"].astype(str).str.strip().any():
            df["year"] = [m.get("year", "") for m in parsed]
        print(f"[Auto-Enrich] Extracted ticker/year from doc_id for {len(df)} samples.")

    # 3. Infer section label from text if missing
    has_section = (
        "section" in df.columns and df["section"].astype(str).str.strip().any()
    )
    if not has_section and "text" in df.columns:
        df["section"] = df["text"].astype(str).apply(infer_section_label)

    return df


def prepare_training_data(
    teacher_data: pd.DataFrame,
    data_prep_params: Dict[str, Any],
    training_params: Optional[Dict[str, Any]] = None,
    schema_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, torch.Tensor], Any]:
    """
    Converts teacher-labeled KG triplets into tokenized PyTorch span/relation tensors.
    Supports either modular namespaced params or a single backward-compatible parameters dict.
    """
    # Auto-heal and enrich provenance metadata if missing from teacher output
    teacher_data = _ensure_provenance_metadata(teacher_data, data_prep_params)

    # Extract configs with graceful fallback for unified or namespaced dicts
    if training_params is None:
        training_params = data_prep_params.get("training", data_prep_params)
    if schema_params is None:
        schema_params = data_prep_params.get("schema", {})

    encoder_model_name = training_params.get("encoder_model_name", "bert-base-uncased")
    max_samples = data_prep_params.get("max_samples", 5000)
    max_gt_triples = data_prep_params.get("max_gt_triples", 15)
    max_seq_length = data_prep_params.get("max_seq_length", 128)
    use_context_prefix = data_prep_params.get("use_context_prefix", True)

    tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)

    # Dynamic schema from configuration
    entity_types = schema_params.get(
        "entity_types",
        ["org", "person", "product", "segment", "fin_metric", "risk_factor", "event"],
    )
    relation_types = schema_params.get(
        "relation_types",
        ["has_metric", "produces", "operates_in", "reports_risk", "led_by"],
    )

    ent_to_id = {ent.lower(): i for i, ent in enumerate(entity_types)}
    rel_to_id = {rel.lower(): i for i, rel in enumerate(relation_types)}

    tokenizer.ent_to_id = ent_to_id
    tokenizer.rel_to_id = rel_to_id

    input_ids_list, attention_masks_list = [], []
    gt_relations_list = []
    gt_subj_types_list, gt_obj_types_list = [], []
    gt_subj_spans_list, gt_obj_spans_list = [], []
    metadata_list: list = []

    for _, row in teacher_data.iterrows():
        try:
            triplets = (
                json.loads(row["triples"].replace("'", '"'))
                if isinstance(row["triples"], str)
                else row["triples"]
            )
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue

        if not triplets:
            continue

        text = row["text"]

        # ── Context prefix ──────────────────────────────────────────────────────
        # Build a natural-language prefix from the chunk's provenance metadata
        # so the encoder can ground self-referential language ('we', 'the Company')
        # without requiring architectural changes.
        # Format: "[AAPL | 2024 | Item 7 – MD&A] <original text>"
        if use_context_prefix:
            ticker = str(row.get("ticker", "")).strip()
            year = str(row.get("year", "")).strip()
            section = str(row.get("section", "")).strip()
            prefix_parts = [p for p in [ticker, year, section] if p]
            if prefix_parts:
                text = f"[{' | '.join(prefix_parts)}] {text}"

        encodings = tokenizer(
            text,
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        ids = encodings["input_ids"][0]
        mask = encodings["attention_mask"][0]

        valid_triples = []
        for t in triplets:
            sub = t.get("head", "").strip().lower()
            if sub == "exact company name":
                sub = (
                    "apple inc."
                    if "AAPL" in row.get("doc_id", "")
                    else (
                        "microsoft corp."
                        if "MSFT" in row.get("doc_id", "")
                        else "company"
                    )
                )

            sub_type = t.get("head_type", "").strip().lower()
            rel = t.get("relation", "").strip().lower()
            obj = t.get("tail", "").strip().lower()
            obj_type = t.get("tail_type", "").strip().lower()

            if (
                sub_type not in ent_to_id
                or obj_type not in ent_to_id
                or rel not in rel_to_id
            ):
                continue

            subj_start, subj_end = align_entities_to_tokens(text, sub, tokenizer, ids)
            obj_start, obj_end = align_entities_to_tokens(text, obj, tokenizer, ids)

            if subj_start != -1 and obj_start != -1:
                valid_triples.append(
                    {
                        "relation": rel_to_id[rel],
                        "subj_type": ent_to_id[sub_type],
                        "obj_type": ent_to_id[obj_type],
                        "subj_span": [subj_start, subj_end],
                        "obj_span": [obj_start, obj_end],
                    }
                )

        if not valid_triples:
            continue

        valid_triples = valid_triples[:max_gt_triples]

        rel_tensor = torch.full(
            (max_gt_triples,), len(relation_types), dtype=torch.long
        )
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

        metadata_list.append(
            {
                "doc_id": row.get("doc_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "ticker": row.get("ticker", ""),
                "year": row.get("year", ""),
                "section": row.get("section", ""),
            }
        )

        if len(input_ids_list) >= max_samples:
            break

    print(f"Gathered {len(input_ids_list)} valid financial samples.")

    tensors = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_masks_list),
        "relations": torch.stack(gt_relations_list),
        "subj_types": torch.stack(gt_subj_types_list),
        "obj_types": torch.stack(gt_obj_types_list),
        "subj_spans": torch.stack(gt_subj_spans_list),
        "obj_spans": torch.stack(gt_obj_spans_list),
        # metadata is kept as a plain list of dicts (not a tensor) so it can
        # travel alongside predictions for traceability at inference time.
        "metadata": metadata_list,
    }
    return tensors, tokenizer


def parse_sec_filings(
    data_prep_params: Union[str, Dict[str, Any], None] = None, max_words: int = 1500
) -> pd.DataFrame:
    """
    Parses SEC 10-K PDFs and EDGAR SGML filings into table-aware markdown chunks.
    Accepts either a parameter dictionary (`params:data_prep`) or positional strings.
    """
    if DocumentConverter is None:
        raise ImportError(
            "docling is required to parse SEC filings. Please install it with `pip install docling`."
        )

    if isinstance(data_prep_params, dict):
        raw_data_dir = data_prep_params.get("raw_pdf_dir", "data/01_raw")
        max_words = data_prep_params.get("max_chunk_words", 1500)
        target_items_regex = data_prep_params.get(
            "target_items_regex", r"(item\s+(1|1a|7|7a|8)\.?\s+)"
        )
        stop_items_regex = data_prep_params.get(
            "stop_items_regex", r"(item\s+(9|10|15)\.?\s+|signat(ure|ures)|part\s+iv)"
        )
    else:
        raw_data_dir = (
            data_prep_params if isinstance(data_prep_params, str) else "data/01_raw"
        )
        target_items_regex = r"(item\s+(1|1a|7|7a|8)\.?\s+)"
        stop_items_regex = r"(item\s+(9|10|15)\.?\s+|signat(ure|ures)|part\s+iv)"

    target_pattern = re.compile(target_items_regex, re.IGNORECASE)
    stop_pattern = re.compile(stop_items_regex, re.IGNORECASE)

    converter = DocumentConverter()
    all_chunks = []

    target_dir = Path(raw_data_dir)
    pdf_files = list(target_dir.glob("*.pdf"))
    edgar_files = list(target_dir.rglob("full-submission.txt"))
    all_files = pdf_files + edgar_files

    for file_path in all_files:
        is_edgar_txt = file_path.name == "full-submission.txt"
        doc_id = file_path.parent.parent.parent.name if is_edgar_txt else file_path.name

        print(f"Extracting {doc_id}...")
        target_parse_path = str(file_path)
        tmp_file = None

        if is_edgar_txt:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                html_content = extract_html_from_sgml(content)
                if html_content:
                    tmp_file = tempfile.NamedTemporaryFile(
                        suffix=".html", delete=False, mode="w", encoding="utf-8"
                    )
                    tmp_file.write(html_content)
                    tmp_file.close()
                    target_parse_path = tmp_file.name
            except Exception as e:
                print(f"Skipping {doc_id} due to SGML extraction error: {e}")
                continue

        try:
            doc = converter.convert(target_parse_path).document
        except Exception as e:
            print(f"Skipping {doc_id} due to parse error: {e}")
            if tmp_file:
                os.unlink(tmp_file.name)
            continue

        if tmp_file:
            os.unlink(tmp_file.name)

        current_chunk = ""
        chunk_idx = 0
        in_target_section = False
        current_section_label = ""  # tracks which Item we are currently inside
        doc_meta = parse_doc_metadata(doc_id)

        for item, _ in doc.iterate_items():
            if hasattr(item, "text") and item.text:
                if target_pattern.search(item.text):
                    in_target_section = True
                    # Update the canonical section label whenever we enter a new item
                    current_section_label = infer_section_label(item.text)
                elif stop_pattern.search(item.text):
                    in_target_section = False
                    current_section_label = ""

            if not in_target_section:
                continue

            if item.label == "table":
                if current_chunk.strip():
                    if is_informative_chunk(current_chunk):
                        all_chunks.append(
                            {
                                "doc_id": doc_id,
                                "chunk_id": chunk_idx,
                                "text": current_chunk.strip(),
                                "ticker": doc_meta["ticker"],
                                "year": doc_meta["year"],
                                "section": current_section_label,
                            }
                        )
                        chunk_idx += 1
                    current_chunk = ""

                table_md = item.export_to_markdown(doc)
                table_text = f"[TABLE START]\n{table_md}\n[TABLE END]"
                all_chunks.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": chunk_idx,
                        "text": table_text,
                        "ticker": doc_meta["ticker"],
                        "year": doc_meta["year"],
                        "section": current_section_label,
                    }
                )
                chunk_idx += 1

            elif hasattr(item, "text") and item.text:
                current_chunk += f"{item.text}\n"

                if len(current_chunk.split()) > max_words:
                    if is_informative_chunk(current_chunk):
                        all_chunks.append(
                            {
                                "doc_id": doc_id,
                                "chunk_id": chunk_idx,
                                "text": current_chunk.strip(),
                                "ticker": doc_meta["ticker"],
                                "year": doc_meta["year"],
                                "section": current_section_label,
                            }
                        )
                        chunk_idx += 1
                    current_chunk = ""

        if current_chunk.strip() and is_informative_chunk(current_chunk):
            all_chunks.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_idx,
                    "text": current_chunk.strip(),
                    "ticker": doc_meta["ticker"],
                    "year": doc_meta["year"],
                    "section": current_section_label,
                }
            )

    return pd.DataFrame(all_chunks)
