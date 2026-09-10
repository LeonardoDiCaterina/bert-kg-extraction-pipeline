import pandas as pd
import json
from vllm import LLM, SamplingParams

# FinReflectKG Closed Schema
FIN_SCHEMA = """
Entity Types: ORG, PERSON, COMP, PRODUCT, SEGMENT, FIN_METRIC, RISK_FACTOR, EVENT, REGULATORY_REQUIREMENT, ESG_TOPIC.
Relationship Types: Has_Stake_In, Operates_In, Produces, Impacts, Involved_In, Impacted_By, Discloses, Complies_With, Supplies, Partners_With.
"""

EXTRACTOR_PROMPT = """<|im_start|>system
You are an expert financial analyst extracting Knowledge Graph triples from SEC filings.<|im_end|>
<|im_start|>user
Extract ALL valid relations using ONLY the defined schema.
Output a JSON list of dictionaries with keys: "head", "head_type", "relation", "tail", "tail_type".

Example Text: "Apple Inc. released the new iPhone 15."
Example Output: [{{ "head": "Apple Inc.", "head_type": "ORG", "relation": "Produces", "tail": "iPhone 15", "tail_type": "PRODUCT" }}]

Schema: {schema}
Text: {text}
JSON Output:<|im_end|>
<|im_start|>assistant
"""

CRITIC_PROMPT = """<|im_start|>system
You are a strict financial data auditor.<|im_end|>
<|im_start|>user
Review the extracted triples against the text and schema.
Flag the following errors:
1. Abstract pronouns (e.g., "The Company", "We", "It") instead of exact company names.
2. Hallucinated relationships not present in the text.
3. Entities/relations that do not strictly match the Schema.

Schema: {schema}
Text: {text}
Extracted Triples: {triples}

Provide a brief critique detailing any errors. If perfect, output "PASS".<|im_end|>
<|im_start|>assistant
"""

REFINER_PROMPT = """<|im_start|>system
You are a Knowledge Graph refinement agent.<|im_end|>
<|im_start|>user
Fix all errors mentioned by the critic. Replace pronouns with exact entity names from the text.
Output ONLY a valid JSON list of dictionaries. Do not include markdown formatting.
Replace pronouns with the actual company name (e.g., "Apple Inc.")

Text: {text}
Initial Triples: {triples}
Critic Feedback: {critique}
Final JSON Output:<|im_end|>
<|im_start|>assistant
"""

def generate_teacher_triplets(parsed_chunks: pd.DataFrame) -> pd.DataFrame:
    """Uses a 3-Agent Reflection loop on vLLM to extract financial 5-tuples."""
    
    print("Loading Qwen 72B Teacher Model onto H100...")
    llm = LLM(model="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4", tensor_parallel_size=1, max_model_len=4096)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)

    # --- AGENT 1: EXTRACTOR ---
    print(f"Agent 1 (Extractor): Processing {len(parsed_chunks)} chunks...")
    ext_prompts = [EXTRACTOR_PROMPT.format(schema=FIN_SCHEMA, text=row['text']) for _, row in parsed_chunks.iterrows()]
    ext_outputs = [r.outputs[0].text.strip() for r in llm.generate(ext_prompts, sampling_params)]

    # --- AGENT 2: CRITIC ---
    print(f"Agent 2 (Critic): Auditing extractions...")
    crit_prompts = [CRITIC_PROMPT.format(schema=FIN_SCHEMA, text=row['text'], triples=ext_outputs[i]) for i, row in parsed_chunks.iterrows()]
    crit_outputs = [r.outputs[0].text.strip() for r in llm.generate(crit_prompts, sampling_params)]

    # --- AGENT 3: REFINER ---
    print(f"Agent 3 (Refiner): Generating final JSON...")
    ref_prompts = [REFINER_PROMPT.format(text=row['text'], triples=ext_outputs[i], critique=crit_outputs[i]) for i, row in parsed_chunks.iterrows()]
    ref_outputs = [r.outputs[0].text.strip() for r in llm.generate(ref_prompts, sampling_params)]

    extracted_data = []
    for i, raw_output in enumerate(ref_outputs):
        try:
            # Clean residual LLM markdown
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