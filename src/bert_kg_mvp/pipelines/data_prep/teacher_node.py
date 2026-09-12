import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from bert_kg_mvp.utils import clean_json_string, resolve_company_name

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None  # type: ignore[assignment, misc]
    SamplingParams = None  # type: ignore[assignment, misc]

# Default schema fallback
DEFAULT_FIN_SCHEMA = """
Entity Types: ORG, PERSON, PRODUCT, SEGMENT, FIN_METRIC, RISK_FACTOR, EVENT.
Relationship Types: Has_Metric, Produces, Operates_In, Reports_Risk, Led_By.
"""

EXTRACTOR_PROMPT = """<|im_start|>system
You are an expert financial analyst specializing in SEC filings (10-K, 10-Q, 8-K) and knowledge graph extraction. You have deep familiarity with how public companies describe their business segments, products, executives, financial metrics, and risk disclosures. Your task is to convert unstructured filing text into precise, schema-conformant knowledge graph triples.<|im_end|>
<|im_start|>user
The following text is an excerpt from a Form 10-K filing for {company_name}.
Document Context: {context}

Any pronoun or generic reference to "the Company," "we," "our," or similar in this text refers strictly to {company_name} ({context}) unless the text explicitly names a different entity (e.g. a subsidiary or named executive). Do NOT hallucinate relations with entities or timeframes outside this filing context.

Extract ALL valid relations from the text below using ONLY the entity types and relationship types defined in the schema. Do not invent new entity or relationship types under any circumstances.

Guidelines:
- Resolve "the Company," "we," "our," etc. directly to "{company_name}" in the head/tail fields — never leave a pronoun or generic company reference as an entity name.
- "head" and "tail" must be exact entity names (use "{company_name}" for the primary entity; use full names as given in the text for all other entities).
- Every triple's "relation" must be one of the defined relationship types, and must connect entity types that make logical sense together.
- Do NOT extract standalone numbers, dates, or dollar amounts as entities. If a number is relevant, it should be captured as an attribute/descriptor of a FIN_METRIC or EVENT entity (e.g. tail = "Net Revenue" not "$12.4 billion").
- Do NOT extract generic or vague nouns (e.g. "the segment," "this metric," "the risk") as head or tail entities — resolve them to their specific named entity if possible, or omit the triple.
- If the same fact appears multiple times with different phrasing, extract it once.
- If no valid relations exist in the text, output an empty JSON list: [].
- Output strictly a JSON list of dictionaries with keys: "head", "head_type", "relation", "tail", "tail_type". No prose, no markdown code fences, no explanation — JSON only.

Example Text: "Apple Inc. released the new iPhone 15."
Example Output: [{{"head": "Apple Inc.", "head_type": "ORG", "relation": "Produces", "tail": "iPhone 15", "tail_type": "PRODUCT"}}]

Schema: {schema}
Document Context: {context}
Company: {company_name}
Text: {text}
JSON Output:<|im_end|>
<|im_start|>assistant
"""

CRITIC_PROMPT = """<|im_start|>system
You are a strict financial data auditor responsible for quality-controlling knowledge graphs extracted from SEC filings. You have zero tolerance for schema violations, hallucinated entities, or triples that misrepresent the source text. Your critiques are used directly by a downstream refinement agent, so they must be specific and actionable.<|im_end|>
<|im_start|>user
The primary entity for this document is {company_name} (Document Context: {context}). Any triple describing "the Company," "we," "our," etc. should already have been resolved to "{company_name}" — treat an unresolved pronoun, generic reference, or hallucinated out-of-context entity left in a head/tail field as a violation.

Review the extracted triples against the source text, filing context, and schema. For each triple, check the following failure modes and flag every violation found, quoting the offending triple exactly:

1. **Invalid head/tail entity**: The head or tail is NOT a recognized Company, Person, Product, Segment, Financial Metric, Risk Factor, or Event as named in the text.
2. **Raw value as node**: A raw number, percentage, date, or dollar amount is extracted as a standalone entity node rather than as part of a properly named FIN_METRIC or EVENT.
3. **Disconnected from primary entity**: The relationship does not explicitly and traceably link back to {company_name}.
4. **Unresolved pronoun**: The head or tail is "the Company," "we," "our," "it," or similar rather than "{company_name}".
5. **Contextual Hallucination**: The triple introduces entities, years, or corporate relationships that contradict the document context {context}.
6. **Type mismatch**: The head_type or tail_type does not match the entity as used.
7. **Unsupported claim**: The triple asserts something not actually stated in the text.
8. **Duplicate**: Redundant representation of the same fact.
9. **Missed extraction**: A clearly valid, schema-conformant relation is present in the text but missing from the extracted triples.

For each issue found, state: (a) the exact triple in question, (b) which failure mode it violates, and (c) a concrete suggested fix.

Schema: {schema}
Document Context: {context}
Company: {company_name}
Text: {text}
Extracted Triples: {triples}

If every triple is fully valid and no relations are missing, output exactly "PASS" and nothing else. Otherwise, provide your critique as a structured list of issues.<|im_end|>
<|im_start|>assistant
"""

REFINER_PROMPT = """<|im_start|>system
You are a Knowledge Graph refinement agent responsible for producing the final, clean set of triples for the FinReflectKG pipeline. You take the critic's feedback as ground truth and apply every correction precisely, without introducing new errors or deviating from the schema.<|im_end|>
<|im_start|>user
The primary entity for this document is {company_name} (Document Context: {context}). Using the critic's feedback, correct the initial triples according to these rules:

- Remove any triple flagged as invalid, hallucinated, duplicate, or containing a raw number/date/dollar amount as a standalone entity.
- Correct any type mismatches identified by the critic.
- Replace all pronouns and vague references with "{company_name}".
- Add any missed valid relations the critic identified.
- Preserve all triples the critic did not flag, unchanged.
- If the critic's feedback was "PASS", return the initial triples exactly as given.

Output ONLY a valid JSON list of dictionaries with keys: "head", "head_type", "relation", "tail", "tail_type". Do not include markdown formatting, code fences, or any explanatory text — JSON only.

Document Context: {context}
Company: {company_name}
Text: {text}
Initial Triples: {triples}
Critic Feedback: {critique}
Final JSON Output:<|im_end|>
<|im_start|>assistant
"""


def get_company_name(doc_id: str, company_map: Optional[Dict[str, str]] = None) -> str:
    """Backward-compatible wrapper around resolve_company_name."""
    return resolve_company_name(doc_id, company_map=company_map)


def format_schema_prompt(schema_params: Optional[Dict[str, Any]]) -> str:
    """Constructs dynamic schema text for LLM prompts from schema configuration."""
    if not schema_params:
        return DEFAULT_FIN_SCHEMA

    entity_types = schema_params.get("entity_types", [])
    relation_types = schema_params.get("relation_types", [])

    if not entity_types or not relation_types:
        return DEFAULT_FIN_SCHEMA

    ent_str = ", ".join(e.upper() for e in entity_types)
    rel_str = ", ".join(r.replace("_", " ").title().replace(" ", "_") for r in relation_types)

    return f"\nEntity Types: {ent_str}.\nRelationship Types: {rel_str}.\n"


def _format_chunk_context(row: pd.Series, company_name: str) -> str:
    """Builds a grounded filing context string: [Ticker | Year | Section]."""
    ticker = str(row.get("ticker", "")).strip()
    year = str(row.get("year", "")).strip()
    section = str(row.get("section", "")).strip()
    parts = [p for p in [ticker, year, section] if p]
    if parts:
        return f"[{' | '.join(parts)}]"
    return f"[{company_name}]"


def _save_checkpoint(data: Any, path: Path) -> None:
    """Safely saves intermediate agent outputs or dataframes to disk in Parquet or CSV format."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame({"output": data})
        try:
            df.to_parquet(path, index=False)
        except Exception:
            df.to_csv(path.with_suffix(".csv"), index=False)
        print(f"[Teacher Pipeline] Checkpoint saved: {path} ({len(df)} rows)")
    except Exception as e:
        print(f"[Teacher Pipeline] Warning: Could not save checkpoint to {path}: {e}")


def _load_checkpoint(path: Path, expected_len: int) -> Optional[List[str]]:
    """Loads intermediate agent outputs from disk if the row count matches expected_len."""
    candidates = [path, path.with_suffix(".csv")]
    for candidate in candidates:
        if candidate.exists():
            try:
                df = pd.read_parquet(candidate) if candidate.suffix == ".parquet" else pd.read_csv(candidate)
                if "output" in df.columns and len(df) == expected_len:
                    print(f"[Teacher Pipeline] Resumed from checkpoint: {candidate} ({len(df)} rows)")
                    return df["output"].fillna("").astype(str).tolist()
                elif len(df) != expected_len:
                    print(
                        f"[Teacher Pipeline] Checkpoint {candidate} row count ({len(df)}) does not match "
                        f"current input length ({expected_len}). Skipping."
                    )
            except Exception as e:
                print(f"[Teacher Pipeline] Notice: Failed to load checkpoint {candidate}: {e}")
    return None


def generate_teacher_triplets(
    parsed_chunks: pd.DataFrame,
    teacher_params: Optional[Dict[str, Any]] = None,
    schema_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Orchestrates the 3-agent LLM distillation pipeline (Extractor -> Critic -> Refiner).
    Parameters are dynamically loaded from `params:teacher` and `params:schema`.

    Resilience & Grounding features:
    - Micro-batch streaming (`batch_size`): Persists completed batches of triples immediately to disk
      so GPU work is never lost if interrupted.
    - Context prefix grounding: Injects [Ticker | Year | Section] into every agent prompt to prevent
      hallucination, exactly matching the student model's document grounding.
    - Intermediate checkpointing for Agent 1, Agent 2, and Agent 3.
    - Defensive prompt clamping in Agent 3 to guarantee total context stays within max_model_len.
    - Optional stratified subsampling via `max_samples` parameter.
    """
    if LLM is None or SamplingParams is None:
        raise ImportError(
            "vllm is required to run the teacher model. Please install it with `pip install vllm`."
        )

    params = teacher_params or {}
    model_name = params.get("model_name", "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4")
    tp_size = params.get("tensor_parallel_size", 1)
    max_model_len = params.get("max_model_len", 8192)
    gpu_memory_utilization = params.get("gpu_memory_utilization", 0.90)
    temperature = params.get("temperature", 0.1)
    max_tokens = params.get("max_tokens", 1024)
    company_map = params.get("company_map", None)

    schema_text = format_schema_prompt(schema_params)
    max_chunk_chars = params.get("max_chunk_chars", 8000)

    # Subsampling configuration
    max_samples = params.get("max_samples", None)
    if max_samples is not None and 0 < int(max_samples) < len(parsed_chunks):
        target_n = int(max_samples)
        sample_random_state = params.get("sample_random_state", 42)
        print(f"[Teacher Pipeline] Subsampling {target_n} chunks (from {len(parsed_chunks)} total) for teacher distillation...")
        if "ticker" in parsed_chunks.columns and parsed_chunks["ticker"].nunique() > 1:
            sampled = (
                parsed_chunks.groupby("ticker", group_keys=False)
                .apply(
                    lambda g: g.sample(
                        min(len(g), max(1, int(round(len(g) * target_n / len(parsed_chunks))))),
                        random_state=sample_random_state,
                    )
                )
            )
            if len(sampled) > target_n:
                sampled = sampled.head(target_n)
            elif len(sampled) < target_n:
                remainder = parsed_chunks[~parsed_chunks.index.isin(sampled.index)]
                fill = remainder.sample(min(len(remainder), target_n - len(sampled)), random_state=sample_random_state)
                sampled = pd.concat([sampled, fill], ignore_index=True)
            parsed_chunks = sampled.reset_index(drop=True)
        else:
            parsed_chunks = parsed_chunks.sample(n=target_n, random_state=sample_random_state).reset_index(drop=True)

    # Checkpoint configuration
    checkpoint_dir_str = params.get("checkpoint_dir", "data/02_intermediate")
    checkpoint_dir = Path(checkpoint_dir_str) if checkpoint_dir_str else None
    resume_checkpoints = params.get("resume_checkpoints", True)

    batches_dir = checkpoint_dir / "teacher_batches" if checkpoint_dir else None
    if batches_dir:
        batches_dir.mkdir(parents=True, exist_ok=True)

    agent1_path = checkpoint_dir / "teacher_agent1_extractions.parquet" if checkpoint_dir else None
    agent2_path = checkpoint_dir / "teacher_agent2_critiques.parquet" if checkpoint_dir else None

    # Prompt safety limits for Agent 3 (Refiner)
    max_ref_text_chars = params.get("max_refiner_text_chars", 4000)
    max_ref_triples_chars = params.get("max_refiner_triples_chars", 2500)
    max_ref_critique_chars = params.get("max_refiner_critique_chars", 2500)

    # Micro-batching configuration: chunk dataset into streaming batches
    batch_size_param = params.get("batch_size", 500)
    batch_size = int(batch_size_param) if (batch_size_param and int(batch_size_param) > 0) else len(parsed_chunks)
    num_batches = math.ceil(len(parsed_chunks) / max(1, batch_size))

    llm = None
    sampling_params = None

    def _get_llm():
        nonlocal llm, sampling_params
        if llm is None:
            print(f"Loading Teacher Model ({model_name})...")
            llm_kwargs: Dict[str, Any] = {
                "model": model_name,
                "tensor_parallel_size": tp_size,
                "max_model_len": max_model_len,
            }
            if gpu_memory_utilization is not None:
                llm_kwargs["gpu_memory_utilization"] = float(gpu_memory_utilization)
            llm = LLM(**llm_kwargs)
            sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        return llm, sampling_params

    all_batch_results: List[pd.DataFrame] = []

    for b_idx in range(num_batches):
        b_start = b_idx * batch_size
        b_end = min(b_start + batch_size, len(parsed_chunks))
        sub_chunks = parsed_chunks.iloc[b_start:b_end].reset_index(drop=True)

        batch_file = batches_dir / f"batch_{b_idx:04d}.parquet" if batches_dir else None

        # Check if entire batch was previously completed
        if resume_checkpoints and batch_file and batch_file.exists():
            try:
                cached_batch = pd.read_parquet(batch_file)
                if len(cached_batch) == len(sub_chunks):
                    print(f"[Teacher Pipeline] Loaded Batch {b_idx + 1}/{num_batches} ({len(cached_batch)} chunks) from cache.")
                    all_batch_results.append(cached_batch)
                    continue
            except Exception as exc:
                print(f"[Teacher Pipeline] Notice: Could not read cached batch {batch_file}: {exc}")

        # Compute context and texts for current batch
        sub_companies = [resolve_company_name(row["doc_id"], company_map=company_map) for _, row in sub_chunks.iterrows()]
        sub_contexts = [_format_chunk_context(row, sub_companies[i]) for i, (_, row) in enumerate(sub_chunks.iterrows())]
        sub_texts = [
            str(row["text"])[:max_chunk_chars] if len(str(row["text"])) > max_chunk_chars else str(row["text"])
            for _, row in sub_chunks.iterrows()
        ]

        # Check for monolithic Agent 1 & Agent 2 checkpoints if running single batch
        ext_outputs = None
        crit_outputs = None
        if num_batches == 1 and resume_checkpoints:
            ext_outputs = _load_checkpoint(agent1_path, len(sub_chunks)) if agent1_path else None
            crit_outputs = _load_checkpoint(agent2_path, len(sub_chunks)) if agent2_path else None

        # --- AGENT 1: EXTRACTOR ---
        if ext_outputs is None:
            engine, s_params = _get_llm()
            print(f"Agent 1 (Extractor) [Batch {b_idx + 1}/{num_batches}]: Processing {len(sub_chunks)} chunks...")
            ext_prompts = [
                EXTRACTOR_PROMPT.format(
                    schema=schema_text,
                    context=sub_contexts[i],
                    company_name=sub_companies[i],
                    text=sub_texts[i]
                )
                for i in range(len(sub_chunks))
            ]
            ext_outputs = [r.outputs[0].text.strip() for r in engine.generate(ext_prompts, s_params)]
            if num_batches == 1 and agent1_path:
                _save_checkpoint(ext_outputs, agent1_path)

        # --- AGENT 2: CRITIC ---
        if crit_outputs is None:
            engine, s_params = _get_llm()
            print(f"Agent 2 (Critic) [Batch {b_idx + 1}/{num_batches}]: Auditing {len(sub_chunks)} chunks...")
            crit_prompts = [
                CRITIC_PROMPT.format(
                    schema=schema_text,
                    context=sub_contexts[i],
                    company_name=sub_companies[i],
                    text=sub_texts[i],
                    triples=ext_outputs[i]
                )
                for i in range(len(sub_chunks))
            ]
            crit_outputs = [r.outputs[0].text.strip() for r in engine.generate(crit_prompts, s_params)]
            if num_batches == 1 and agent2_path:
                _save_checkpoint(crit_outputs, agent2_path)

        # --- AGENT 3: REFINER ---
        engine, s_params = _get_llm()
        print(f"Agent 3 (Refiner) [Batch {b_idx + 1}/{num_batches}]: Generating final JSON...")
        ref_prompts = [
            REFINER_PROMPT.format(
                context=sub_contexts[i],
                company_name=sub_companies[i],
                text=sub_texts[i][:max_ref_text_chars],
                triples=ext_outputs[i][:max_ref_triples_chars],
                critique=crit_outputs[i][:max_ref_critique_chars],
            )
            for i in range(len(sub_chunks))
        ]
        ref_outputs = [r.outputs[0].text.strip() for r in engine.generate(ref_prompts, s_params)]

        batch_extracted_data = []
        for i, raw_output in enumerate(ref_outputs):
            clean_json = clean_json_string(raw_output)
            try:
                triples = json.loads(clean_json)
            except json.JSONDecodeError:
                triples = []

            batch_extracted_data.append({
                "doc_id":   sub_chunks.iloc[i]["doc_id"],
                "chunk_id": sub_chunks.iloc[i]["chunk_id"],
                "ticker":   sub_chunks.iloc[i].get("ticker",  ""),
                "year":     sub_chunks.iloc[i].get("year",    ""),
                "section":  sub_chunks.iloc[i].get("section", ""),
                "text":     sub_chunks.iloc[i]["text"],
                "triples":  triples,
            })

        batch_result_df = pd.DataFrame(batch_extracted_data)
        if batch_file:
            _save_checkpoint(batch_result_df, batch_file)

        all_batch_results.append(batch_result_df)

        # Incrementally persist partial dataset to disk after every batch
        if checkpoint_dir:
            try:
                rolling_df = pd.concat(all_batch_results, ignore_index=True)
                partial_path = checkpoint_dir / "teacher_extracted_triplets_partial.csv"
                rolling_df.to_csv(partial_path, index=False)
                tot_triples = sum(len(row.get("triples", [])) for _, row in rolling_df.iterrows())
                print(
                    f"[Teacher Pipeline] Progress: Batch {b_idx + 1}/{num_batches} persisted to disk "
                    f"({len(rolling_df)} cumulative chunks, {tot_triples} triples extracted)."
                )
            except Exception as exc:
                print(f"[Teacher Pipeline] Notice: Could not write rolling progress: {exc}")

    final_df = pd.concat(all_batch_results, ignore_index=True) if all_batch_results else pd.DataFrame()
    return final_df