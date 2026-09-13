from unittest.mock import MagicMock, patch
import torch
from bert_kg_mvp.utils import (
    align_entities_to_tokens,
    clean_json_string,
    extract_html_from_sgml,
    extract_rebel_triplets,
    infer_section_label,
    is_informative_chunk,
    parse_doc_metadata,
    parse_triplet_string,
    resolve_company_name,
)


def test_is_informative_chunk():
    # Less than 50 words
    assert not is_informative_chunk("Too short.")
    # Boilerplate check mark phrase
    assert not is_informative_chunk("Word " * 60 + "Indicate by check mark if...")
    # Legitimate chunk
    assert is_informative_chunk(
        "Word " * 60 + "Apple Inc. designs semiconductors and operating systems."
    )


def test_extract_rebel_triplets():
    raw_str = "<s> <triplet> Microsoft <subj> Windows <obj> develops </s>"
    triplets = extract_rebel_triplets(raw_str)
    assert len(triplets) == 1
    assert triplets[0] == {"head": "Microsoft", "type": "develops", "tail": "Windows"}


def test_parse_triplet_string():
    sample = (
        "<triplet> Apple Inc. <subj_type> ORG <relation> produces <obj> iPhone <obj_type> PRODUCT "
        "<triplet> Tim Cook <subj_type> PERSON <relation> led_by <obj> Apple Inc. <obj_type> ORG"
    )
    result = parse_triplet_string(sample)
    assert len(result) == 2
    assert ("Apple Inc.", "produces", "iPhone") in result
    assert ("Tim Cook", "led_by", "Apple Inc.") in result


def test_clean_json_string():
    raw_json = '```json\n[{"head": "A", "tail": "B"}]\n```'
    assert clean_json_string(raw_json) == '[{"head": "A", "tail": "B"}]'

    raw_generic = "```\n[1, 2, 3]\n```"
    assert clean_json_string(raw_generic) == "[1, 2, 3]"

    already_clean = '{"key": "val"}'
    assert clean_json_string(already_clean) == '{"key": "val"}'


def test_align_entities_to_tokens():
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda s, add_special_tokens=False: {
        "apple": [101, 102],
        "nonexistent": [999],
        "": [],
    }.get(s, [500])

    input_ids = torch.tensor([1, 2, 101, 102, 3, 4])
    start, end = align_entities_to_tokens("text", "apple", mock_tokenizer, input_ids)
    assert start == 2
    assert end == 3

    # Test BPE/Fast Tokenizer Character Mapping fallback
    mock_tokenizer.is_fast = True
    mock_encodings = MagicMock()
    # Mock char_to_token mapping for "apple inc."
    mock_encodings.char_to_token = MagicMock(side_effect=lambda b, c: 5 if 10 <= c <= 19 else None)
    
    text = "We love Apple Inc. products"
    # "Apple Inc." starts at char index 8
    # Oh wait, my mocked char_to_token above is naive. Let's just mock it specifically for the start/end bounds.
    
    def _mock_char_to_token(batch, char_idx):
        if char_idx == 8: return 4
        if char_idx == 17: return 5
        return None
        
    mock_encodings.char_to_token.side_effect = _mock_char_to_token
    s_fast, e_fast = align_entities_to_tokens(text, "Apple Inc.", mock_tokenizer, input_ids, encodings=mock_encodings)
    assert s_fast == 4
    assert e_fast == 5

    # Non-existent entity
    s_miss, e_miss = align_entities_to_tokens(
        "text", "nonexistent", mock_tokenizer, input_ids
    )
    assert s_miss == -1
    assert e_miss == -1

    # Empty entity
    s_empty, e_empty = align_entities_to_tokens("text", "", mock_tokenizer, input_ids)
    assert s_empty == -1
    assert e_empty == -1


def test_augmented_kg_dataset():
    from bert_kg_mvp.utils.dataset import AugmentedKGDataset
    import random
    
    base_tensors = {
        "input_ids": torch.tensor([[101, 10, 11, 12, 13, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
        "relations": torch.tensor([[2]]),
        "subj_types": torch.tensor([[1]]),
        "obj_types": torch.tensor([[3]]),
        "subj_spans": torch.tensor([[[1, 2]]]),
        "obj_spans": torch.tensor([[[3, 4]]]),
    }
    
    tokenizer = MagicMock()
    tokenizer.mask_token_id = 999
    
    # 1. Test Deterministic Eval mode (no augmentation applied)
    ds_eval = AugmentedKGDataset(base_tensors, mask_token_id=999, pad_token_id=0, no_relation_idx=5, mask_prob=0.0, prefix_drop_prob=0.0, span_jitter_prob=0.0)
    item_eval = ds_eval[0]
    assert torch.equal(item_eval[0], base_tensors["input_ids"][0])
    
    # 2. Test Entity Masking (mask_prob=1.0)
    random.seed(42)
    ds_mask = AugmentedKGDataset(base_tensors, mask_token_id=999, pad_token_id=0, no_relation_idx=5, mask_prob=1.0, prefix_drop_prob=0.0, span_jitter_prob=0.0)
    
    # Force random to trigger masking
    with patch("random.random", return_value=0.01):
        item_mask = ds_mask[0]
        # Since p=1.0 and random=0.01, it masks BOTH subj and obj!
        assert item_mask["input_ids"][1] == 999
        assert item_mask["input_ids"][2] == 999
        assert item_mask["input_ids"][3] == 999
        assert item_mask["input_ids"][4] == 999
        
    # 3. Test Prefix Dropping (p_drop_prefix=1.0)
    ds_drop = AugmentedKGDataset(base_tensors, tokenizer, is_training=True, p_mask=0.0, p_drop_prefix=1.0, p_jitter=0.0)
    with patch("random.random", return_value=0.01):
        # Suppose prefix is just token [10]. Drop it!
        with patch("bert_kg_mvp.utils.dataset.AugmentedKGDataset._drop_prefix") as mock_drop:
            mock_drop.return_value = (torch.tensor([101, 11, 12, 13, 102, 0]), torch.tensor([1, 1, 1, 1, 1, 0]), torch.tensor([[0, 1]]), torch.tensor([[2, 3]]))
            item_drop = ds_drop[0]
            mock_drop.assert_called_once()
            assert item_drop["input_ids"][-1] == 0 # Padding added
            
    # 4. Test Span Jittering
    ds_jitter = AugmentedKGDataset(base_tensors, tokenizer, is_training=True, p_mask=0.0, p_drop_prefix=0.0, p_jitter=1.0)
    with patch("random.random", return_value=0.01):
        with patch("random.choice", return_value=1): # Shift right by 1
            item_jitter = ds_jitter[0]
            assert item_jitter["subj_spans"][0][0].item() == 2 # 1+1
            assert item_jitter["subj_spans"][0][1].item() == 3 # 2+1



def test_extract_html_from_sgml():
    sgml_valid = "<SUBMISSION>\n<DOCUMENT>\n<TYPE>10-K\n<TEXT>\n<html><body>10-K Financial Text</body></html>\n</TEXT>\n</DOCUMENT>\n</SUBMISSION>"
    extracted = extract_html_from_sgml(sgml_valid)
    assert extracted == "<html><body>10-K Financial Text</body></html>"

    # Missing document tags
    assert extract_html_from_sgml("Just plain text without tags") is None
    # Missing text tags
    assert extract_html_from_sgml("<DOCUMENT>No text tag</DOCUMENT>") is None


def test_resolve_company_name():
    # Default map
    assert resolve_company_name("AAPL_2024.pdf") == "Apple Inc."
    assert resolve_company_name("MSFT_filing") == "Microsoft Corp."
    assert resolve_company_name("unknown_doc") == "The Corporation"

    # Custom map
    custom = {"GOOG": "Google LLC"}
    assert resolve_company_name("GOOG_10K.txt", company_map=custom) == "Google LLC"
    assert resolve_company_name("AAPL_10K.txt", company_map=custom) == "The Corporation"


def test_infer_section_label():
    assert infer_section_label("Item 1. Business") == "Item 1 – Business"
    assert infer_section_label("Item 1A. Risk Factors") == "Item 1A – Risk Factors"
    assert (
        infer_section_label("ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS")
        == "Item 7 – MD&A"
    )
    assert (
        infer_section_label("Item 7A. Quantitative Disclosures")
        == "Item 7A – Market Risk"
    )
    assert (
        infer_section_label("Item 8. Financial Statements")
        == "Item 8 – Financial Statements"
    )
    assert infer_section_label("Item 9B. Other Information") == "Unknown"
    assert infer_section_label("Just random text") == "Unknown"


def test_parse_doc_metadata():
    # SEC EDGAR folder structure
    path1 = "sec-edgar-filings/AAPL/10-K/0000320193-24-000106/full-submission.txt"
    meta1 = parse_doc_metadata(path1)
    assert meta1["ticker"] == "AAPL"
    assert meta1["year"] == "2024"

    # Flat filename
    path2 = "MSFT_2023_10K.pdf"
    meta2 = parse_doc_metadata(path2)
    assert meta2["ticker"] == "MSFT"
    assert meta2["year"] == "2023"

    # Without year or ticker
    meta3 = parse_doc_metadata("unknown_document.txt")
    assert meta3["year"] == ""
