from unittest.mock import MagicMock
import torch
from bert_kg_mvp.utils import (
    align_entities_to_tokens,
    clean_json_string,
    extract_html_from_sgml,
    extract_rebel_triplets,
    is_informative_chunk,
    parse_triplet_string,
    resolve_company_name,
)


def test_is_informative_chunk():
    # Less than 50 words
    assert not is_informative_chunk("Too short.")
    # Boilerplate check mark phrase
    assert not is_informative_chunk("Word " * 60 + "Indicate by check mark if...")
    # Legitimate chunk
    assert is_informative_chunk("Word " * 60 + "Apple Inc. designs semiconductors and operating systems.")


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
    raw_json = "```json\n[{\"head\": \"A\", \"tail\": \"B\"}]\n```"
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
        "": []
    }.get(s, [500])

    input_ids = torch.tensor([1, 2, 101, 102, 3, 4])
    start, end = align_entities_to_tokens("text", "apple", mock_tokenizer, input_ids)
    assert start == 2
    assert end == 3

    # Non-existent entity
    s_miss, e_miss = align_entities_to_tokens("text", "nonexistent", mock_tokenizer, input_ids)
    assert s_miss == -1
    assert e_miss == -1

    # Empty entity
    s_empty, e_empty = align_entities_to_tokens("text", "", mock_tokenizer, input_ids)
    assert s_empty == -1
    assert e_empty == -1


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
