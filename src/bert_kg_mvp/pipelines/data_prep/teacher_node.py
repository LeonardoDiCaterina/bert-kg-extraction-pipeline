import pandas as pd
import json
from vllm import LLM, SamplingParams

# FinReflectKG Closed Schema
FIN_SCHEMA = """
Entity Types: ORG, PERSON, COMP, PRODUCT, SEGMENT, FIN_METRIC, RISK_FACTOR, EVENT, REGULATORY_REQUIREMENT, ESG_TOPIC.
Relationship Types: Has_Stake_In, Operates_In, Produces, Impacts, Involved_In, Impacted_By, Discloses, Complies_With, Supplies, Partners_With.
"""

PROMPT_TEMPLATE = """You are an expert financial analyst extracting Knowledge Graph triples from SEC 10-K filings.
Extract ALL valid relations using ONLY the defined schema. 
Replace abstract pronouns ("we", "the company") with the exact entity name (e.g. "Apple Inc.").
Return a JSON list of dictionaries with keys: "head", "head_type", "relation", "tail", "tail_type".

Schema:
{schema}

Text:
{text}

JSON Output:"""

def generate_teacher_triplets(parsed_chunks: pd.DataFrame) -> pd.DataFrame:
    """Uses vLLM and a 72B Teacher model to extract financial 5-tuples."""
    
    print("Loading Qwen 72B Teacher Model onto H100...")
    # Load 4-bit quantized model to fit in 80GB VRAM
    llm = LLM(model="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4", tensor_parallel_size=1, max_model_len=4096)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)

    prompts = []
    for _, row in parsed_chunks.iterrows():
        prompt = PROMPT_TEMPLATE.format(schema=FIN_SCHEMA, text=row['text'])
        prompts.append(prompt)

    print(f"Executing Batch Extraction on {len(prompts)} chunks...")
    responses = llm.generate(prompts, sampling_params)

    extracted_data = []
    for i, response in enumerate(responses):
        raw_output = response.outputs[0].text.strip()
        
        # Fallback parsing for LLM JSON output
        try:
            # Strip markdown code blocks if the LLM generates them
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
