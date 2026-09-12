import json
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
The following text is an excerpt from a filing by {company_name}. Any pronoun or generic reference to "the Company," "we," "our," or similar in this text refers to {company_name} unless the text explicitly names a different entity (e.g. a subsidiary or named executive).

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
Example Output: [{{ "head": "Apple Inc.", "head_type": "ORG", "relation": "Produces", "tail": "iPhone 15", "tail_type": "PRODUCT" }}]

Schema: {schema}
Company: {company_name}
Text: {text}
JSON Output:<|im_end|>
<|im_start|>assistant
"""

CRITIC_PROMPT = """<|im_start|>system
You are a strict financial data auditor responsible for quality-controlling knowledge graphs extracted from SEC filings. You have zero tolerance for schema violations, hallucinated entities, or triples that misrepresent the source text. Your critiques are used directly by a downstream refinement agent, so they must be specific and actionable.<|im_end|>
<|im_start|>user
The primary entity for this document is {company_name}. Any triple describing "the Company," "we," "our," etc. should already have been resolved to "{company_name}" — treat an unresolved pronoun or generic reference left in a head/tail field as a violation.

Review the extracted triples against the source text and the schema. For each triple, check the following failure modes and flag every violation found, quoting the offending triple exactly:

1. **Invalid head/tail entity**: The head or tail is NOT a recognized Company, Person, Product, Segment, Financial Metric, Risk Factor, or Event as named in the text.
2. **Raw value as node**: A raw number, percentage, date, or dollar amount is extracted as a standalone entity node rather than as part of a properly named FIN_METRIC or EVENT.
3. **Disconnected from primary entity**: The relationship does not explicitly and traceably link back to {company_name}.
4. **Unresolved pronoun**: The head or tail is "the Company," "we," "our," "it," or similar rather than "{company_name}".
5. **Type mismatch**: The head_type or tail_type does not match the entity as used.
6. **Unsupported claim**: The triple asserts something not actually stated in the text.
7. **Duplicate**: Redundant representation of the same fact.
8. **Missed extraction**: A clearly valid, schema-conformant relation is present in the text but missing from the extracted triples.

For each issue found, state: (a) the exact triple in question, (b) which failure mode it violates, and (c) a concrete suggested fix.

Schema: {schema}
Company: {company_name}
Text: {text}
Extracted Triples: {triples}

If every triple is fully valid and no relations are missing, output exactly "PASS" and nothing else. Otherwise, provide your critique as a structured list of issues.<|im_end|>
<|im_start|>assistant
"""

REFINER_PROMPT = """<|im_start|>system
You are a Knowledge Graph refinement agent responsible for producing the final, clean set of triples for the FinReflectKG pipeline. You take the critic's feedback as ground truth and apply every correction precisely, without introducing new errors or deviating from the schema.<|im_end|>
<|im_start|>user
The primary entity for this document is {company_name}. Using the critic's feedback, correct the initial triples according to these rules:

- Remove any triple flagged as invalid, hallucinated, duplicate, or containing a raw number/date/dollar amount as a standalone entity.
- Correct any type mismatches identified by the critic.
- Replace all pronouns and vague references with "{company_name}".
- Add any missed valid relations the critic identified.
- Preserve all triples the critic did not flag, unchanged.
- If the critic's feedback was "PASS", return the initial triples exactly as given.

Output ONLY a valid JSON list of dictionaries with keys: "head", "head_type", "relation", "tail", "tail_type". Do not include markdown formatting, code fences, or any explanatory text — JSON only.

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


def _save_checkpoint(data: List[str], path: Path) -> None:
    """Safely saves intermediate agent outputs to disk in Parquet or CSV format."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"output": data})
        try:
            df.to_parquet(path, index=False)
        except Exception:
            df.to_csv(path.with_suffix(".csv"), index=False)
        print(f"[Teacher Pipeline] Checkpoint saved: {path} ({len(data)} rows)")
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

    Resilience features:
    - Intermediate checkpointing for Agent 1, Agent 2, and Agent 3 to prevent GPU compute loss.
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

    agent1_path = checkpoint_dir / "teacher_agent1_extractions.parquet" if checkpoint_dir else None
    agent2_path = checkpoint_dir / "teacher_agent2_critiques.parquet" if checkpoint_dir else None
    agent3_path = checkpoint_dir / "teacher_agent3_refinements.parquet" if checkpoint_dir else None

    # Prompt safety limits for Agent 3 (Refiner)
    max_ref_text_chars = params.get("max_refiner_text_chars", 4000)
    max_ref_triples_chars = params.get("max_refiner_triples_chars", 2500)
    max_ref_critique_chars = params.get("max_refiner_critique_chars", 2500)

    # Pre-check if all stages are already checkpointed
    cached_ref_outputs = (
        _load_checkpoint(agent3_path, len(parsed_chunks))
        if (resume_checkpoints and agent3_path)
        else None
    )

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

    companies = [resolve_company_name(row["doc_id"], company_map=company_map) for _, row in parsed_chunks.iterrows()]
    # Defensively truncate oversized chunks to prevent context overflow
    chunk_texts = [
        str(row["text"])[:max_chunk_chars] if len(str(row["text"])) > max_chunk_chars else str(row["text"])
        for _, row in parsed_chunks.iterrows()
    ]

    # --- AGENT 1: EXTRACTOR ---
    ext_outputs = (
        _load_checkpoint(agent1_path, len(parsed_chunks))
        if (resume_checkpoints and agent1_path)
        else None
    )
    if ext_outputs is None:
        engine, s_params = _get_llm()
        print(f"Agent 1 (Extractor): Processing {len(parsed_chunks)} chunks...")
        ext_prompts = [
            EXTRACTOR_PROMPT.format(schema=schema_text, company_name=companies[i], text=chunk_texts[i])
            for i in range(len(parsed_chunks))
        ]
        ext_outputs = [r.outputs[0].text.strip() for r in engine.generate(ext_prompts, s_params)]
        if agent1_path:
            _save_checkpoint(ext_outputs, agent1_path)
    else:
        print(f"Agent 1 (Extractor): Skipped (loaded {len(ext_outputs)} outputs from checkpoint).")

    # --- AGENT 2: CRITIC ---
    crit_outputs = (
        _load_checkpoint(agent2_path, len(parsed_chunks))
        if (resume_checkpoints and agent2_path)
        else None
    )
    if crit_outputs is None:
        engine, s_params = _get_llm()
        print(f"Agent 2 (Critic): Auditing extractions for {len(parsed_chunks)} chunks...")
        crit_prompts = [
            CRITIC_PROMPT.format(schema=schema_text, company_name=companies[i], text=chunk_texts[i], triples=ext_outputs[i])
            for i in range(len(parsed_chunks))
        ]
        crit_outputs = [r.outputs[0].text.strip() for r in engine.generate(crit_prompts, s_params)]
        if agent2_path:
            _save_checkpoint(crit_outputs, agent2_path)
    else:
        print(f"Agent 2 (Critic): Skipped (loaded {len(crit_outputs)} outputs from checkpoint).")

    # --- AGENT 3: REFINER ---
    ref_outputs = cached_ref_outputs
    if ref_outputs is None:
        engine, s_params = _get_llm()
        print(f"Agent 3 (Refiner): Generating final JSON for {len(parsed_chunks)} chunks...")
        ref_prompts = [
            REFINER_PROMPT.format(
                company_name=companies[i],
                text=chunk_texts[i][:max_ref_text_chars],
                triples=ext_outputs[i][:max_ref_triples_chars],
                critique=crit_outputs[i][:max_ref_critique_chars],
            )
            for i in range(len(parsed_chunks))
        ]
        ref_outputs = [r.outputs[0].text.strip() for r in engine.generate(ref_prompts, s_params)]
        if agent3_path:
            _save_checkpoint(ref_outputs, agent3_path)
    else:
        print(f"Agent 3 (Refiner): Skipped (loaded {len(ref_outputs)} outputs from checkpoint).")

    extracted_data = []
    for i, raw_output in enumerate(ref_outputs):
        clean_json = clean_json_string(raw_output)
        try:
            triples = json.loads(clean_json)
        except json.JSONDecodeError:
            triples = []

        extracted_data.append({
            "doc_id":   parsed_chunks.iloc[i]["doc_id"],
            "chunk_id": parsed_chunks.iloc[i]["chunk_id"],
            "ticker":   parsed_chunks.iloc[i].get("ticker",  ""),
            "year":     parsed_chunks.iloc[i].get("year",    ""),
            "section":  parsed_chunks.iloc[i].get("section", ""),
            "text":     parsed_chunks.iloc[i]["text"],
            "triples":  triples,
        })

    return pd.DataFrame(extracted_data)