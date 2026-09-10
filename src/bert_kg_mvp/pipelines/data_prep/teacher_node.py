import pandas as pd
import json
from vllm import LLM, SamplingParams

FIN_SCHEMA = """
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
- Every triple's "relation" must be one of the five defined relationship types, and must connect entity types that make logical sense together (e.g. Has_Metric should connect an ORG or SEGMENT to a FIN_METRIC, not two FIN_METRIC nodes).
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

1. **Invalid head/tail entity**: The head or tail is NOT a recognized Company, Person, Product, Segment, Financial Metric, Risk Factor, or Event as named in the text (e.g., forbidding vague or fragment entities like "Balance," "2009 issuance," "the Company's operations," or "increase").
2. **Raw value as node**: A raw number, percentage, date, or dollar amount is extracted as a standalone entity node rather than as part of a properly named FIN_METRIC or EVENT (e.g., tail = "$3.2 million" is invalid; tail = "Operating Income" is valid).
3. **Disconnected from primary entity**: The relationship does not explicitly and traceably link back to {company_name}, either directly or through a valid intermediate entity (e.g., a SEGMENT or PRODUCT that itself belongs to {company_name}).
4. **Unresolved pronoun**: The head or tail is "the Company," "we," "our," "it," or similar rather than "{company_name}" or another explicit named entity.
5. **Type mismatch**: The head_type or tail_type does not match the entity as used, or the relation type does not logically fit the head/tail entity types per the schema (e.g., Led_By should link PERSON to ORG/SEGMENT, not FIN_METRIC to PERSON).
6. **Unsupported claim**: The triple asserts something not actually stated or clearly implied in the text (i.e., a hallucination).
7. **Duplicate**: The same fact is represented more than once with redundant triples.
8. **Missed extraction**: A clearly valid, schema-conformant relation is present in the text but missing from the extracted triples.

For each issue found, state: (a) the exact triple in question, (b) which failure mode it violates, and (c) a concrete suggested fix (e.g., correct entity name, correct type, or "remove this triple").

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
- Correct any type mismatches (head_type, tail_type, or relation) identified by the critic.
- Replace all pronouns and vague references (e.g., "the Company," "we," "our," "it," "the segment") with "{company_name}" (for the primary entity) or the exact, fully-resolved entity name as it appears in the text (for any other entity, e.g. "iPhone Segment"). Never use "the Company" or a pronoun in the final output.
- Add any missed valid relations the critic identified, formatted identically to the other triples.
- Do not reintroduce any error type the critic flagged, even in slightly different form.
- Preserve all triples the critic did not flag, unchanged.
- If the critic's feedback was "PASS", return the initial triples exactly as given, with any remaining pronouns still resolved to "{company_name}" or the explicit entity name.

Output ONLY a valid JSON list of dictionaries with keys: "head", "head_type", "relation", "tail", "tail_type". Do not include markdown formatting, code fences, or any explanatory text — JSON only.

Company: {company_name}
Text: {text}
Initial Triples: {triples}
Critic Feedback: {critique}
Final JSON Output:<|im_end|>
<|im_start|>assistant
"""

def get_company_name(doc_id: str) -> str:
    doc_upper = doc_id.upper()
    if "AAPL" in doc_upper:
        return "Apple Inc."
    elif "MSFT" in doc_upper:
        return "Microsoft Corp."
    elif "AMZN" in doc_upper:
        return "Amazon.com, Inc."
    return "The Corporation"

def generate_teacher_triplets(parsed_chunks: pd.DataFrame) -> pd.DataFrame:
    print("Loading Qwen 72B Teacher Model onto H100...")
    llm = LLM(model="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4", tensor_parallel_size=1, max_model_len=4096)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)

    # Pre-calculate company mappings for each row
    companies = [get_company_name(row["doc_id"]) for _, row in parsed_chunks.iterrows()]

    # --- AGENT 1: EXTRACTOR ---
    print(f"Agent 1 (Extractor): Processing {len(parsed_chunks)} chunks...")
    ext_prompts = [
        EXTRACTOR_PROMPT.format(schema=FIN_SCHEMA, company_name=companies[i], text=row['text']) 
        for i, row in parsed_chunks.iterrows()
    ]
    ext_outputs = [r.outputs[0].text.strip() for r in llm.generate(ext_prompts, sampling_params)]

    # --- AGENT 2: CRITIC ---
    print(f"Agent 2 (Critic): Auditing extractions...")
    crit_prompts = [
        CRITIC_PROMPT.format(schema=FIN_SCHEMA, company_name=companies[i], text=row['text'], triples=ext_outputs[i]) 
        for i, row in parsed_chunks.iterrows()
    ]
    crit_outputs = [r.outputs[0].text.strip() for r in llm.generate(crit_prompts, sampling_params)]

    # --- AGENT 3: REFINER ---
    print(f"Agent 3 (Refiner): Generating final JSON...")
    ref_prompts = [
        REFINER_PROMPT.format(company_name=companies[i], text=row['text'], triples=ext_outputs[i], critique=crit_outputs[i]) 
        for i, row in parsed_chunks.iterrows()
    ]
    ref_outputs = [r.outputs[0].text.strip() for r in llm.generate(ref_prompts, sampling_params)]

    extracted_data = []
    for i, raw_output in enumerate(ref_outputs):
        try:
            clean_json = raw_output.replace("```json", "").replace("```", "").strip()
            triples = json.loads(clean_json)
        except json.JSONDecodeError:
            triples = []
            
        extracted_data.append({
            "chunk_id": parsed_chunks.iloc[i]["chunk_id"],
            "text": parsed_chunks.iloc[i]["text"],
            "triples": triples
        })
        
    return pd.DataFrame(extracted_data)