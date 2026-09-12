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
Entity Types: ORG, PERSON, COMP, PRODUCT, SEGMENT, FIN_METRIC, RISK_FACTOR, EVENT, REGULATORY_REQUIREMENT, ESG_TOPIC.
Relationship Types: Has_Metric, Produces, Operates_In, Reports_Risk, Led_By, Has_Stake_In, Impacts, Involved_In, Impacted_By, Discloses, Complies_With, Supplies, Partners_With.

Semantic Signatures & Allowed Relations:
- Has_Metric: [ORG or SEGMENT] -> [FIN_METRIC] (Named financial metric, ratio, or accounting line item; tail must be metric name, never a raw number, dollar amount, or full sentence)
- Produces: [ORG, SEGMENT or COMP] -> [PRODUCT] (Commercial product, software platform, hardware, online service)
- Operates_In: [ORG or COMP] -> [SEGMENT] (Reportable business segment, division, or geographic operating unit)
- Reports_Risk: [ORG or SEGMENT] -> [RISK_FACTOR] (Explicit operational, competitive, macroeconomic, or market risk factor; concise noun phrase of 1 to 5 words)
- Led_By: [ORG, SEGMENT or COMP] -> [PERSON] (Named executive officer, director, or board member)
- Has_Stake_In: [ORG or COMP] -> [ORG or COMP] (Indicates full or partial ownership or equity interest)
- Impacts: [EVENT, RISK_FACTOR, REGULATORY_REQUIREMENT or ESG_TOPIC] -> [FIN_METRIC, ORG or SEGMENT] (Specifies the broad influence or effect an entity or event has on financial performance, market trends, or other key outcomes)
- Involved_In: [ORG, PERSON or COMP] -> [EVENT] (Specifies direct involvement in an event such as a merger, acquisition, or litigation)
- Impacted_By: [ORG, SEGMENT or FIN_METRIC] -> [EVENT or RISK_FACTOR] (Indicates that the entity was materially affected by a major event)
- Discloses: [ORG] -> [ESG_TOPIC, FIN_METRIC or RISK_FACTOR] (Reveals or reports information, metrics, or developments)
- Complies_With: [ORG or COMP] -> [REGULATORY_REQUIREMENT] (Meets regulatory or policy requirements)
- Supplies: [COMP] -> [ORG or COMP] (Indicates vendor or supplier relationship)
- Partners_With: [ORG or COMP] -> [ORG or COMP] (Indicates formal or strategic collaboration)
"""

EXTRACTOR_PROMPT = """<|im_start|>system
You are a principal financial knowledge engineer specializing in SEC Form 10-K filings and knowledge graph extraction. Your goal is to produce high-precision, schema-conformant training data to train a downstream neural student model equipped with extractive span pointer heads.

================================================================================
CRITICAL ONTOLOGY & TYPE CONSTRAINTS (ZERO DEVIATION PERMITTED)
================================================================================
Entity Types (STRICTLY LIMITED TO THESE 10 LABELS):
1. ORG: Filing Company (Issuer: The public company that is the subject of the 10-K filing) (FORBIDDEN: "Company", "Corporation", "Firm").
2. PERSON: Key individuals (e.g., CEO, CFO, Board members).
3. COMP: External companies referenced in the filing, including competitors, suppliers, customers, or partners.
4. PRODUCT: Products or services offered by the company or competitors (e.g., iPhone, AWS) (FORBIDDEN: "Program", "Offering").
5. SEGMENT: Internal divisions or business segments of the filer ORG (e.g., Cloud segment, North America retail) (FORBIDDEN: "Division", "Unit").
6. FIN_METRIC: Financial metrics or values (e.g., Net Income, EBITDA, CapEx, Revenue) (FORBIDDEN: "Metric", "KPI", "Financial Metric", "Description").
7. RISK_FACTOR: Documented risks (e.g., market risk, supply chain risk, regulatory risk) (FORBIDDEN: "Risk", "Issue").
8. EVENT: Material events such as pandemics, natural disasters, M&A events, regulatory changes.
9. REGULATORY_REQUIREMENT: Specific regulations or legal frameworks (e.g., Basel III, GDPR, SEC requirements).
10. ESG_TOPIC: Environmental, Social, and Governance themes (e.g., Carbon Emissions, DEI, Climate Risk).

Relationship Types & Semantic Signatures (STRICTLY LIMITED TO THESE 13 LABELS):
1. Has_Metric: (head: ORG | SEGMENT, tail: FIN_METRIC)
   - Company or segment tracks, reports, or values a named financial metric.
   - Tail MUST be the metric name (1-5 words), NEVER a dollar amount, percentage, date, or full sentence.
2. Produces: (head: ORG | SEGMENT | COMP, tail: PRODUCT)
   - Entity develops, manufactures, sells, or delivers a product/platform.
3. Operates_In: (head: ORG | COMP, tail: SEGMENT)
   - Entity operates or reports revenue under a business segment or division.
4. Reports_Risk: (head: ORG | SEGMENT, tail: RISK_FACTOR)
   - Entity identifies an explicit risk factor. Tail MUST be a concise noun phrase (1-5 words), NEVER a sentence or narrative.
5. Led_By: (head: ORG | SEGMENT | COMP, tail: PERSON)
   - Entity is led or managed by an executive or director.
6. Has_Stake_In: (head: ORG | COMP, tail: ORG | COMP)
   - Indicates full or partial ownership or equity interest.
7. Impacts: (head: EVENT | RISK_FACTOR | REGULATORY_REQUIREMENT | ESG_TOPIC, tail: FIN_METRIC | ORG | SEGMENT)
   - Specifies the broad influence or effect an entity or event has on financial performance, market trends, or other key outcomes.
8. Involved_In: (head: ORG | PERSON | COMP, tail: EVENT)
   - Specifies direct involvement in an event such as a merger, acquisition, or litigation.
9. Impacted_By: (head: ORG | SEGMENT | FIN_METRIC, tail: EVENT | RISK_FACTOR)
   - Indicates that the entity was materially affected by a major event.
10. Discloses: (head: ORG, tail: ESG_TOPIC | FIN_METRIC | RISK_FACTOR)
   - Reveals or reports information, metrics, or developments.
11. Complies_With: (head: ORG | COMP, tail: REGULATORY_REQUIREMENT)
   - Meets regulatory or policy requirements.
12. Supplies: (head: COMP, tail: ORG | COMP)
   - Indicates vendor or supplier relationship.
13. Partners_With: (head: ORG | COMP, tail: ORG | COMP)
   - Indicates formal or strategic collaboration.

================================================================================
GROUNDING & SPAN RULES FOR NEURAL STUDENT EXTRACTION
================================================================================
1. VERBATIM EXTRACTIVE SPANS:
   - The student model aligns token spans directly to the source text. Every "tail" entity MUST appear VERBATIM in the text.
   - Do NOT paraphrase, summarize, or normalize entity strings (e.g. if the text says "implied volatility", extract "implied volatility", not "Expected Implied Volatility Metric").
2. CONCISE NOUN PHRASES (STRICTLY 1 TO 5 WORDS):
   - Entities must be tight noun phrases (e.g. "employee stock purchase rights", "supply chain disruptions", "gross margin").
   - NEVER extract full sentences, clauses, parentheticals, or paragraphs as entities!
3. COMPANY GROUNDING & PRONOUN RESOLUTION:
   - If the text uses first-person pronouns ("we", "our", "us") or generic phrases ("the Company", "the Corporation"), resolve the head entity to "{company_name}" or the ticker in Document Context {context}.
   - NEVER output "The Corporation", "The Company", "we", or "our" as an entity name.
4. NO STANDALONE NUMBERS, PERCENTAGES, OR DATES:
   - Never extract "$1.5 billion", "12%", "2024", or "$500M" as an entity. The financial metric itself is the FIN_METRIC.
5. NEGATIVE EXTRACTION:
   - If the text does not explicitly state any of the allowed relations, return an empty JSON list: []. Do not invent relations.
6. OUTPUT FORMAT:
   - Output strictly a JSON list of dictionaries with keys: "head", "head_type", "relation", "tail", "tail_type". No markdown, no explanations.

================================================================================
FEW-SHOT EXTRACTION EXAMPLES
================================================================================
Example 1 (Financial Metrics & Valuation Assumptions):
Text: "[CSCO | 2024 | Item 8] The valuation of employee stock purchase rights and the related assumptions are for the employee stock purchases made during the respective fiscal years. We used implied volatility for traded options on our stock as the expected volatility assumption in the Black-Scholes model."
Output:
[
  {{"head": "CSCO", "head_type": "ORG", "relation": "Has_Metric", "tail": "employee stock purchase rights", "tail_type": "FIN_METRIC"}},
  {{"head": "CSCO", "head_type": "ORG", "relation": "Has_Metric", "tail": "implied volatility", "tail_type": "FIN_METRIC"}},
  {{"head": "CSCO", "head_type": "ORG", "relation": "Has_Metric", "tail": "Black-Scholes model", "tail_type": "FIN_METRIC"}}
]

Example 2 (Risk Disclosure):
Text: "[TSLA | 2023 | Item 1A] We are subject to risks associated with supply chain disruptions and semiconductor shortages that could adversely affect our manufacturing operations."
Output:
[
  {{"head": "TSLA", "head_type": "ORG", "relation": "Reports_Risk", "tail": "supply chain disruptions", "tail_type": "RISK_FACTOR"}},
  {{"head": "TSLA", "head_type": "ORG", "relation": "Reports_Risk", "tail": "semiconductor shortages", "tail_type": "RISK_FACTOR"}}
]

Example 3 (Segments, Products & Leadership):
Text: "[MSFT | 2024 | Item 1] Microsoft Corp. operates in three segments: Productivity and Business Processes, Intelligent Cloud, and More Personal Computing. The Intelligent Cloud segment produces Azure. Satya Nadella serves as Chief Executive Officer."
Output:
[
  {{"head": "Microsoft Corp.", "head_type": "ORG", "relation": "Operates_In", "tail": "Productivity and Business Processes", "tail_type": "SEGMENT"}},
  {{"head": "Microsoft Corp.", "head_type": "ORG", "relation": "Operates_In", "tail": "Intelligent Cloud", "tail_type": "SEGMENT"}},
  {{"head": "Microsoft Corp.", "head_type": "ORG", "relation": "Operates_In", "tail": "More Personal Computing", "tail_type": "SEGMENT"}},
  {{"head": "Intelligent Cloud", "head_type": "SEGMENT", "relation": "Produces", "tail": "Azure", "tail_type": "PRODUCT"}},
  {{"head": "Microsoft Corp.", "head_type": "ORG", "relation": "Led_By", "tail": "Satya Nadella", "tail_type": "PERSON"}}
]

Example 4 (Negative Example - Boilerplate/No Relations):
Text: "[AAPL | 2024 | Part I] Table of Contents. Item 1. Business. Item 1A. Risk Factors. Item 1B. Unresolved Staff Comments."
Output:
[]

================================================================================
FORBIDDEN COUNTER-EXAMPLES (DO NOT REPEAT THESE DEFECTS):
- FORBIDDEN: [{{"head": "The Corporation", "head_type": "Company", "relation": "Reports_Risk", "tail": "The valuation of employee stock purchase rights and the related assumptions are for the employee stock purchases made during the respective fiscal years.", "tail_type": "Description"}}]
  * Violations: "Company" and "Description" are illegal types; tail is a 24-word sentence; relation is not a risk; head is unresolved pronoun/generic.
- FORBIDDEN: [{{"head": "Apple Inc.", "head_type": "ORG", "relation": "Has_Metric", "tail": "$89.5 billion", "tail_type": "FIN_METRIC"}}]
  * Violation: Raw dollar figures and percentages are not entities. Extract the metric name ("net sales").
<|im_end|>
<|im_start|>user
Document Context: {context}
Filing Company: {company_name}
Schema: {schema}

Filing Text:
\"\"\"
{text}
\"\"\"

Extract all valid, concise knowledge graph triples from the filing text conforming strictly to the ontology and extraction rules.
JSON Output:<|im_end|>
<|im_start|>assistant
"""

CRITIC_PROMPT = """<|im_start|>system
You are the Lead Knowledge Graph Auditor for the FinReflectKG pipeline. Your task is to perform an uncompromising quality audit on triples extracted by the Extractor agent from SEC Form 10-K filings.

Any defect in these triples will degrade downstream neural student model training. You must detect every violation and provide explicit, actionable corrections.

SCHEMA REFERENCE:
- Entity Types: ORG, PERSON, COMP, PRODUCT, SEGMENT, FIN_METRIC, RISK_FACTOR, EVENT, REGULATORY_REQUIREMENT, ESG_TOPIC
- Relationship Types: Has_Metric, Produces, Operates_In, Reports_Risk, Led_By, Has_Stake_In, Impacts, Involved_In, Impacted_By, Discloses, Complies_With, Supplies, Partners_With
- Semantic Signatures:
  * Has_Metric: [ORG | SEGMENT] -> [FIN_METRIC]
  * Produces: [ORG | SEGMENT | COMP] -> [PRODUCT]
  * Operates_In: [ORG | COMP] -> [SEGMENT]
  * Reports_Risk: [ORG | SEGMENT] -> [RISK_FACTOR]
  * Led_By: [ORG | SEGMENT | COMP] -> [PERSON]
  * Has_Stake_In: [ORG | COMP] -> [ORG | COMP]
  * Impacts: [EVENT | RISK_FACTOR | REGULATORY_REQUIREMENT | ESG_TOPIC] -> [FIN_METRIC | ORG | SEGMENT]
  * Involved_In: [ORG | PERSON | COMP] -> [EVENT]
  * Impacted_By: [ORG | SEGMENT | FIN_METRIC] -> [EVENT | RISK_FACTOR]
  * Discloses: [ORG] -> [ESG_TOPIC | FIN_METRIC | RISK_FACTOR]
  * Complies_With: [ORG | COMP] -> [REGULATORY_REQUIREMENT]
  * Supplies: [COMP] -> [ORG | COMP]
  * Partners_With: [ORG | COMP] -> [ORG | COMP]<|im_end|>
<|im_start|>user
Document Context: {context}
Filing Company: {company_name}
Schema: {schema}

Filing Text:
\"\"\"
{text}
\"\"\"

Extracted Triples:
{triples}

Audit each triple against the filing text and ontology schema. Check for these specific failure modes:

1. **Non-Canonical Entity Type**: head_type or tail_type is NOT strictly one of the allowed schema types (e.g. using "Company", "Metric", "Description", "Program", "Service", "Division", "KPI"). Must be converted to canonical schema type.
2. **Invalid Relation Type or Semantic Signature Mismatch**: The relation is not one of the allowed relations, or violates argument types (e.g. Has_Metric pointing to something other than FIN_METRIC, Reports_Risk pointing to an accounting narrative or non-RISK_FACTOR).
3. **Oversized Entity (Sentence as Node)**: head or tail is a full sentence, clause, narrative, or exceeds 5 words. Quote the exact concise 1-4 word core noun phrase to replace it.
4. **Non-Verbatim / Hallucinated Span**: The tail (or head) does not appear verbatim in the source text or context prefix.
5. **Standalone Number or Date**: Tail is a raw number, dollar amount, percentage, or date (e.g. "$5.2B", "15%").
6. **Unresolved Pronoun**: head is "we", "our", "the Company", or "The Corporation" instead of "{company_name}" or the ticker in {context}.
7. **Redundant / Duplicate**: Multiple triples representing the exact same fact.
8. **Missed Valid Relation**: A clear relation stated in the text was omitted.

If all triples are 100% compliant, concise, verbatim, and correct, output exactly "PASS".
Otherwise, list each issue with:
- Offending Triple
- Defect Type
- Exact Actionable Fix (specifying canonical types and shortened noun phrases)<|im_end|>
<|im_start|>assistant
"""

REFINER_PROMPT = """<|im_start|>system
You are the Final Knowledge Graph Refiner for the FinReflectKG pipeline. Your mission is to take the Extractor's initial triples and the Critic's audit feedback, and output the definitive, schema-perfect JSON dataset of knowledge graph triples.

HARD REFINEMENT DIRECTIVES:
1. STRICT TYPE CANONICALIZATION:
   - Convert all non-schema types to the canonical types:
     * "Company", "Corporation", "Firm" -> "ORG"
     * "Metric", "KPI", "Financial Metric", "Description" -> "FIN_METRIC"
     * "Program", "Service", "System", "Offering" -> "PRODUCT"
     * "Division", "Unit" -> "SEGMENT"
     * "Risk", "Threat", "Issue" -> "RISK_FACTOR"
   - Discard any triple whose types cannot be mapped to the canonical types: ORG, PERSON, COMP, PRODUCT, SEGMENT, FIN_METRIC, RISK_FACTOR, EVENT, REGULATORY_REQUIREMENT, ESG_TOPIC.
2. ENFORCE VALID RELATIONS:
   - Every relation MUST be strictly one of: Has_Metric, Produces, Operates_In, Reports_Risk, Led_By, Has_Stake_In, Impacts, Involved_In, Impacted_By, Discloses, Complies_With, Supplies, Partners_With.
   - Discard any triple with an unapproved relation.
3. SHORTEN OVERSIZED ENTITIES:
   - Replace any full sentence, clause, or multi-word narrative with its concise core noun phrase (1 to 5 words) appearing verbatim in the text.
   - Example: "The valuation of employee stock purchase rights and the related assumptions..." -> "employee stock purchase rights"
   - Example: "Implied volatility for traded options on our stock as the expected volatility assumption..." -> "implied volatility"
4. RESOLVE COMPANY REFERENCES:
   - Replace "we", "our", "the Company", "The Corporation" with "{company_name}" or the ticker from {context}.
5. REMOVE NUMERIC STANDALONES & UNGROUNDED CLAUSES:
   - Delete any triple whose entity is a raw number, percentage, date, or dollar amount.
   - Delete any triple where the tail entity does not appear in the filing text.
6. APPLY CRITIC ADDITIONS & CORRECTIONS:
   - Incorporate any valid missed relations identified by the Critic.
   - If the Critic output was "PASS", return the initial triples cleanly formatted.
7. PURE JSON OUTPUT:
   - Output ONLY a valid JSON list of dictionaries with keys: "head", "head_type", "relation", "tail", "tail_type". No markdown code blocks, no conversational preamble.<|im_end|>
<|im_start|>user
Document Context: {context}
Filing Company: {company_name}
Schema: {schema}

Filing Text:
\"\"\"
{text}
\"\"\"

Initial Triples:
{triples}

Critic Feedback:
{critique}

Output the final, refined JSON list of triples:
JSON Output:<|im_end|>
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
    rel_str = ", ".join(
        r.replace("_", " ").title().replace(" ", "_") for r in relation_types
    )

    return (
        f"\nEntity Types: {ent_str}.\n"
        f"Relationship Types: {rel_str}.\n\n"
        "Semantic Signatures & Allowed Relations:\n"
        "- Has_Metric: [ORG or SEGMENT] -> [FIN_METRIC] (Named financial metric, ratio, or accounting line item; tail must be metric name, never a raw number, dollar amount, or full sentence)\n"
        "- Produces: [ORG or SEGMENT] -> [PRODUCT] (Commercial product, software platform, hardware, online service)\n"
        "- Operates_In: [ORG] -> [SEGMENT] (Reportable business segment, division, or geographic operating unit)\n"
        "- Reports_Risk: [ORG or SEGMENT] -> [RISK_FACTOR] (Explicit operational, competitive, macroeconomic, or market risk factor; concise noun phrase of 1 to 5 words)\n"
        "- Led_By: [ORG or SEGMENT] -> [PERSON] (Named executive officer, director, or board member)\n"
    )


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
                df = (
                    pd.read_parquet(candidate)
                    if candidate.suffix == ".parquet"
                    else pd.read_csv(candidate)
                )
                if "output" in df.columns and len(df) == expected_len:
                    print(
                        f"[Teacher Pipeline] Resumed from checkpoint: {candidate} ({len(df)} rows)"
                    )
                    return df["output"].fillna("").astype(str).tolist()
                elif len(df) != expected_len:
                    print(
                        f"[Teacher Pipeline] Checkpoint {candidate} row count ({len(df)}) does not match "
                        f"current input length ({expected_len}). Skipping."
                    )
            except Exception as e:
                print(
                    f"[Teacher Pipeline] Notice: Failed to load checkpoint {candidate}: {e}"
                )
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

    # 0. Ensure provenance metadata (ticker, year, section) is populated
    from bert_kg_mvp.pipelines.data_prep.nodes import _ensure_provenance_metadata
    from bert_kg_mvp.utils import is_table_skeleton

    parsed_chunks = _ensure_provenance_metadata(parsed_chunks, params)

    # Filter out empty tables and pure markdown skeletons
    skeleton_mask = parsed_chunks["text"].astype(str).apply(is_table_skeleton)
    if skeleton_mask.any():
        dropped_count = int(skeleton_mask.sum())
        print(
            f"[Teacher Pipeline] Filtered out {dropped_count} uninformative table skeleton chunks before distillation."
        )
        parsed_chunks = parsed_chunks[~skeleton_mask].reset_index(drop=True)

    # Subsampling configuration
    max_samples = params.get("max_samples", None)
    if max_samples is not None and 0 < int(max_samples) < len(parsed_chunks):
        target_n = int(max_samples)
        sample_random_state = params.get("sample_random_state", 42)
        print(
            f"[Teacher Pipeline] Subsampling {target_n} chunks (from {len(parsed_chunks)} total) for teacher distillation..."
        )
        if "ticker" in parsed_chunks.columns and parsed_chunks["ticker"].nunique() > 1:
            sampled = parsed_chunks.groupby("ticker", group_keys=False).apply(
                lambda g: g.sample(
                    min(
                        len(g),
                        max(1, int(round(len(g) * target_n / len(parsed_chunks)))),
                    ),
                    random_state=sample_random_state,
                )
            )
            if len(sampled) > target_n:
                sampled = sampled.head(target_n)
            elif len(sampled) < target_n:
                remainder = parsed_chunks[~parsed_chunks.index.isin(sampled.index)]
                fill = remainder.sample(
                    min(len(remainder), target_n - len(sampled)),
                    random_state=sample_random_state,
                )
                sampled = pd.concat([sampled, fill], ignore_index=True)
            parsed_chunks = sampled.reset_index(drop=True)
        else:
            parsed_chunks = parsed_chunks.sample(
                n=target_n, random_state=sample_random_state
            ).reset_index(drop=True)

    # Checkpoint configuration
    checkpoint_dir_str = params.get("checkpoint_dir", "data/02_intermediate")
    checkpoint_dir = Path(checkpoint_dir_str) if checkpoint_dir_str else None
    resume_checkpoints = params.get("resume_checkpoints", True)

    batches_dir = checkpoint_dir / "teacher_batches" if checkpoint_dir else None
    if batches_dir:
        batches_dir.mkdir(parents=True, exist_ok=True)

    agent1_path = (
        checkpoint_dir / "teacher_agent1_extractions.parquet"
        if checkpoint_dir
        else None
    )
    agent2_path = (
        checkpoint_dir / "teacher_agent2_critiques.parquet" if checkpoint_dir else None
    )

    # Prompt safety limits for Agent 3 (Refiner)
    max_ref_text_chars = params.get("max_refiner_text_chars", 4000)
    max_ref_triples_chars = params.get("max_refiner_triples_chars", 2500)
    max_ref_critique_chars = params.get("max_refiner_critique_chars", 2500)

    # Micro-batching configuration: chunk dataset into streaming batches
    batch_size_param = params.get("batch_size", 500)
    batch_size = (
        int(batch_size_param)
        if (batch_size_param and int(batch_size_param) > 0)
        else len(parsed_chunks)
    )
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
            sampling_params = SamplingParams(
                temperature=temperature, max_tokens=max_tokens
            )
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
                    print(
                        f"[Teacher Pipeline] Loaded Batch {b_idx + 1}/{num_batches} ({len(cached_batch)} chunks) from cache."
                    )
                    all_batch_results.append(cached_batch)
                    continue
            except Exception as exc:
                print(
                    f"[Teacher Pipeline] Notice: Could not read cached batch {batch_file}: {exc}"
                )

        # Compute context and texts for current batch
        sub_companies = []
        for _, row in sub_chunks.iterrows():
            c_name = resolve_company_name(str(row["doc_id"]), company_map=company_map)
            if (c_name == "The Corporation" or not c_name) and row.get("ticker"):
                c_name = str(row["ticker"])
            sub_companies.append(c_name)

        sub_contexts = [
            _format_chunk_context(row, sub_companies[i])
            for i, (_, row) in enumerate(sub_chunks.iterrows())
        ]
        sub_texts = [
            str(row["text"])[:max_chunk_chars]
            if len(str(row["text"])) > max_chunk_chars
            else str(row["text"])
            for _, row in sub_chunks.iterrows()
        ]

        # Check for monolithic Agent 1 & Agent 2 checkpoints if running single batch
        ext_outputs = None
        crit_outputs = None
        if num_batches == 1 and resume_checkpoints:
            ext_outputs = (
                _load_checkpoint(agent1_path, len(sub_chunks)) if agent1_path else None
            )
            crit_outputs = (
                _load_checkpoint(agent2_path, len(sub_chunks)) if agent2_path else None
            )

        # --- AGENT 1: EXTRACTOR ---
        if ext_outputs is None:
            engine, s_params = _get_llm()
            print(
                f"Agent 1 (Extractor) [Batch {b_idx + 1}/{num_batches}]: Processing {len(sub_chunks)} chunks..."
            )
            ext_prompts = [
                EXTRACTOR_PROMPT.format(
                    schema=schema_text,
                    context=sub_contexts[i],
                    company_name=sub_companies[i],
                    text=sub_texts[i],
                )
                for i in range(len(sub_chunks))
            ]
            ext_outputs = [
                r.outputs[0].text.strip()
                for r in engine.generate(ext_prompts, s_params)
            ]
            if num_batches == 1 and agent1_path:
                _save_checkpoint(ext_outputs, agent1_path)

        # --- AGENT 2: CRITIC ---
        if crit_outputs is None:
            engine, s_params = _get_llm()
            print(
                f"Agent 2 (Critic) [Batch {b_idx + 1}/{num_batches}]: Auditing {len(sub_chunks)} chunks..."
            )
            crit_prompts = [
                CRITIC_PROMPT.format(
                    schema=schema_text,
                    context=sub_contexts[i],
                    company_name=sub_companies[i],
                    text=sub_texts[i],
                    triples=ext_outputs[i],
                )
                for i in range(len(sub_chunks))
            ]
            crit_outputs = [
                r.outputs[0].text.strip()
                for r in engine.generate(crit_prompts, s_params)
            ]
            if num_batches == 1 and agent2_path:
                _save_checkpoint(crit_outputs, agent2_path)

        # --- AGENT 3: REFINER ---
        engine, s_params = _get_llm()
        print(
            f"Agent 3 (Refiner) [Batch {b_idx + 1}/{num_batches}]: Generating final JSON..."
        )
        ref_prompts = [
            REFINER_PROMPT.format(
                schema=schema_text,
                context=sub_contexts[i],
                company_name=sub_companies[i],
                text=sub_texts[i][:max_ref_text_chars],
                triples=ext_outputs[i][:max_ref_triples_chars],
                critique=crit_outputs[i][:max_ref_critique_chars],
            )
            for i in range(len(sub_chunks))
        ]
        ref_outputs = [
            r.outputs[0].text.strip() for r in engine.generate(ref_prompts, s_params)
        ]

        batch_extracted_data = []
        for i, raw_output in enumerate(ref_outputs):
            clean_json = clean_json_string(raw_output)
            triples = []
            try:
                triples = json.loads(clean_json)
            except Exception:
                try:
                    import ast

                    triples = ast.literal_eval(clean_json)
                except Exception:
                    triples = []

            if not isinstance(triples, list):
                triples = []

            batch_extracted_data.append(
                {
                    "doc_id": sub_chunks.iloc[i]["doc_id"],
                    "chunk_id": sub_chunks.iloc[i]["chunk_id"],
                    "ticker": sub_chunks.iloc[i].get("ticker", ""),
                    "year": sub_chunks.iloc[i].get("year", ""),
                    "section": sub_chunks.iloc[i].get("section", ""),
                    "text": sub_chunks.iloc[i]["text"],
                    "triples": triples,
                }
            )

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
                tot_triples = sum(
                    len(row.get("triples", [])) for _, row in rolling_df.iterrows()
                )
                print(
                    f"[Teacher Pipeline] Progress: Batch {b_idx + 1}/{num_batches} persisted to disk "
                    f"({len(rolling_df)} cumulative chunks, {tot_triples} triples extracted)."
                )
            except Exception as exc:
                print(
                    f"[Teacher Pipeline] Notice: Could not write rolling progress: {exc}"
                )

    final_df = (
        pd.concat(all_batch_results, ignore_index=True)
        if all_batch_results
        else pd.DataFrame()
    )
    return final_df
