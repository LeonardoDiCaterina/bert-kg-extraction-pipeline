# Kedro Pipeline Modularization, Parameterization & Anti-Pattern Refactoring

This implementation plan addresses the anti-patterns currently present across the Kedro pipelines (`data_prep`, `training`, `inference`), centralizes domain configuration in `conf/base/parameters.yml`, extracts reusable logic into a dedicated `bert_kg_mvp.utils` package, and aligns catalog entries with Kedro best practices.

---

## Anti-Pattern Analysis & Findings

| Area | Current Anti-Pattern | Proposed Solution |
| :--- | :--- | :--- |
| **Schema Fragmentation** | `ENTITY_TYPES` & `RELATION_TYPES` are hardcoded in `data_prep/nodes.py`, prompt strings are hardcoded in `teacher_node.py`, and class counts (`num_relations=5`, `num_ent_types=7`, `valid_gt != 5`) are hardcoded in `training/nodes.py` and `inference/nodes.py`. | Centralize schema in `parameters.yml` under `schema:`. Derive all counts, masks, and head sizes dynamically from this single source of truth. |
| **Monolithic Nodes** | `parse_sec_filings` mixes SEC SGML extraction, regex section filtering, Docling conversion, text windowing, and boilerplate checks. | Move SGML extraction, HTML cleanup, and section heuristics to `bert_kg_mvp.utils.sec_edgar` and `bert_kg_mvp.utils.text_processing`. |
| **Shared Helpers in Node Files** | Helpers like `align_entities_to_tokens`, `is_informative_chunk`, `extract_rebel_triplets`, and `parse_triplet_string` live directly inside node modules or are copy-pasted across `evaluate.py` and `visualize_graph.py`. | Extract into a shared `src/bert_kg_mvp/utils/` package (`text_processing.py`, `tokenization.py`, `sec_edgar.py`). |
| **Global Parameters Anti-Pattern** | Nodes take the giant `parameters: dict` and call `.get(...)` for arbitrary keys, obscuring actual dependencies. | Group parameters by pipeline namespace (`params:data_prep`, `params:teacher`, `params:training`, `params:inference`, `params:schema`) so nodes declare only the configs they need. |
| **Hardcoded Hyperparameters** | Inference sample size (20), batch size (4), teacher model name, Qwen max tokens (1024), temperature (0.1), and company ticker mappings (`AAPL`, `MSFT`, `AMZN`) are hardcoded inside Python code. | Expose all of them in `conf/base/parameters.yml`. |
| **Outdated Standalone Scripts** | `evaluate.py` and `predict.py` still invoke the legacy Architecture 1 causal decoder (`BERTToKnowledgeGraph`) and duplicate `parse_triplet_string`. | Update them to import from `bert_kg_mvp.utils` and use `DynamicKGExtractor`. |

---

## Proposed Changes

### 1. Configuration & Domain Schemas

#### [MODIFY] [parameters.yml](file:///Users/leonardodicaterina/Documents/GitHub/bert_kg_mvp/conf/base/parameters.yml)
Organize parameters into clean, namespaced sections:
- `schema`:
  - `entity_types`: `["org", "person", "product", "segment", "fin_metric", "risk_factor", "event"]`
  - `relation_types`: `["has_metric", "produces", "operates_in", "reports_risk", "led_by"]`
- `data_prep`:
  - `raw_pdf_dir`: `"data/01_raw"`
  - `max_chunk_words`: `1500`
  - `max_seq_length`: `128`
  - `max_gt_triples`: `15`
  - `max_samples`: `5000`
  - `target_items_regex`: `"(item\\s+(1|1a|7|7a|8)\\.?\\s+)"`
  - `stop_items_regex`: `"(item\\s+(9|10|15)\\.?\\s+|signat(ure|ures)|part\\s+iv)"`
- `teacher`:
  - `model_name`: `"Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"`
  - `tensor_parallel_size`: `1`
  - `max_model_len`: `4096`
  - `temperature`: `0.1`
  - `max_tokens`: `1024`
  - `company_map`:
    - `AAPL`: `"Apple Inc."`
    - `MSFT`: `"Microsoft Corp."`
    - `AMZN`: `"Amazon.com, Inc."`
    - `NVDA`: `"NVIDIA Corporation"`
    - `GOOGL`: `"Alphabet Inc."`
    - `META`: `"Meta Platforms, Inc."`
- `training`:
  - `encoder_model_name`: `"bert-base-uncased"`
  - `decoder_num_layers`: `4`
  - `num_queries`: `15`
  - `freeze_strategy`: `"partial"`
  - `unfrozen_top_layers`: `4`
  - `d_model`: `768`
  - `epochs`: `10`
  - `learning_rate`: `0.00005`
  - `batch_size`: `4`
  - `gradient_accumulation_steps`: `8`
- `inference`:
  - `sample_size`: `20`
  - `batch_size`: `4`

#### [MODIFY] [catalog.yml](file:///Users/leonardodicaterina/Documents/GitHub/bert_kg_mvp/conf/base/catalog.yml)
- Clarify dataset names:
  - `kg_inference_metrics`: `data/07_model_output/evaluation_metrics.csv`
  - `kg_predicted_triplets`: `data/07_model_output/predicted_triplets.csv` (optional extraction output)

---

### 2. Utilities Package (`src/bert_kg_mvp/utils/`)

Create modular, single-responsibility utility files:

#### [NEW] [__init__.py](file:///Users/leonardodicaterina/Documents/GitHub/bert_kg_mvp/src/bert_kg_mvp/utils/__init__.py)
Expose standard utility helpers.

#### [NEW] [text_processing.py](file:///Users/leonardodicaterina/Documents/GitHub/bert_kg_mvp/src/bert_kg_mvp/utils/text_processing.py)
- `is_informative_chunk(text: str, min_words: int = 50) -> bool`: Boilerplate and length filter.
- `extract_rebel_triplets(text: str) -> list[dict]`: Parse Rebel tag formatting.
- `parse_triplet_string(text: str) -> set[tuple]`: Extract `(subject, relation, object)` tuples from predicted strings.
- `clean_json_string(text: str) -> str`: Strip markdown code fences (````json ... ````) and prefix text.

#### [NEW] [tokenization.py](file:///Users/leonardodicaterina/Documents/GitHub/bert_kg_mvp/src/bert_kg_mvp/utils/tokenization.py)
- `align_entities_to_tokens(text: str, entity_str: str, tokenizer, input_ids) -> tuple[int, int]`: Tokenizer sliding window matcher for pointer networks.

#### [NEW] [sec_edgar.py](file:///Users/leonardodicaterina/Documents/GitHub/bert_kg_mvp/src/bert_kg_mvp/utils/sec_edgar.py)
- `extract_html_from_sgml(content: str) -> str | None`: Extracts `<DOCUMENT><TEXT>` primary HTML block from SEC EDGAR `full-submission.txt` without crashing Docling.
- `resolve_company_name(doc_id: str, company_map: dict[str, str], default_name: str = "The Corporation") -> str`: Dynamic ticker-to-name resolver.

---

### 3. Pipelines & Nodes Refactoring

#### [MODIFY] [data_prep/nodes.py](file:///Users/leonardodicaterina/Documents/GitHub/bert_kg_mvp/src/bert_kg_mvp/pipelines/data_prep/nodes.py)
- Import helpers from `bert_kg_mvp.utils`.
- Update `parse_sec_filings` to accept `raw_data_dir`, `max_words`, `target_items_regex`, `stop_items_regex`.
- Update `prepare_training_data` to accept `teacher_data`, `params_data_prep`, `params_schema` (or `params:schema` and `params:training`).
- Dynamically build `ent_to_id` and `rel_to_id` directly from `schema.entity_types` and `schema.relation_types`.

#### [MODIFY] [data_prep/teacher_node.py](file:///Users/leonardodicaterina/Documents/GitHub/bert_kg_mvp/src/bert_kg_mvp/pipelines/data_prep/teacher_node.py)
- Dynamically format `FIN_SCHEMA` prompt text from `params_schema` instead of a static hardcoded block.
- Parameterize `model_name`, `temperature`, `max_tokens`, `max_model_len`, and `company_map`.
- Use `clean_json_string` from `bert_kg_mvp.utils.text_processing`.

#### [MODIFY] [training/nodes.py](file:///Users/leonardodicaterina/Documents/GitHub/bert_kg_mvp/src/bert_kg_mvp/pipelines/training/nodes.py)
- Accept `params_training` and `params_schema`.
- Derive `num_relations = len(params_schema["relation_types"])` and `num_ent_types = len(params_schema["entity_types"])`.
- Pass dynamic counts to `DynamicKGExtractor` and `SetCriterion`.

#### [MODIFY] [inference/nodes.py](file:///Users/leonardodicaterina/Documents/GitHub/bert_kg_mvp/src/bert_kg_mvp/pipelines/inference/nodes.py)
- Accept `params_inference` and `params_schema`.
- Parameterize `sample_size` and `batch_size`.
- Replace hardcoded `gt_rels != 5` with `gt_rels != len(params_schema["relation_types"])` (`no_relation` index).
- Remove duplicated `parse_triplet_string` or import from `bert_kg_mvp.utils.text_processing`.

#### [MODIFY] [pipeline.py files](file:///Users/leonardodicaterina/Documents/GitHub/bert_kg_mvp/src/bert_kg_mvp/pipelines/)
- Update Kedro node definitions to cleanly wire namespaced parameters (e.g. `inputs=["parsed_10k_chunks", "params:teacher", "params:schema"]`).

---

### 4. Updating Standalone Scripts & Tests

#### [MODIFY] [evaluate.py](file:///Users/leonardodicaterina/Documents/GitHub/bert_kg_mvp/evaluate.py) & [predict.py](file:///Users/leonardodicaterina/Documents/GitHub/bert_kg_mvp/predict.py)
- Import `parse_triplet_string` from `bert_kg_mvp.utils.text_processing`.
- Modernize inference logic to work with `DynamicKGExtractor` output dict.

#### [MODIFY] [tests/](file:///Users/leonardodicaterina/Documents/GitHub/bert_kg_mvp/tests/)
- Add unit tests for the new `utils/` modules.
- Update pipeline test signatures for the new parameter namespacing.
- Ensure 0 Ruff errors, 0 Mypy errors, and maintain >90% code coverage.

---

## Verification Plan

### Automated Tests
1. **Ruff Linting**:
   ```bash
   ruff check src/ tests/
   ```
2. **Mypy Type Checking**:
   ```bash
   mypy src/ tests/
   ```
3. **Full Pytest Coverage**:
   ```bash
   conda run -n bert_kg_mvp pytest tests/ --cov=src --cov-report=term-missing
   ```
   Verify coverage remains $\ge 90\%$.

4. **Pipeline Dry-run / Structure Validation**:
   ```bash
   conda run -n bert_kg_mvp kedro registry describe
   ```
