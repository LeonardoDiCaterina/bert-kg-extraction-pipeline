import os
import tempfile
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from bert_kg_mvp.pipelines.data_prep.nodes import (
    is_informative_chunk,
    prepare_training_data,
    parse_sec_filings,
    extract_rebel_triplets,
)
from bert_kg_mvp.pipelines.data_prep.pipeline import create_pipeline


def test_is_informative_chunk():
    # Short chunk (<50 words)
    assert not is_informative_chunk("Short snippet of text.")

    # High checkbox / administrative ratio
    checkbox_text = "Item 1. " + ("words " * 60) + "indicate by check mark if registrant is..."
    assert not is_informative_chunk(checkbox_text)

    # Valid informative chunk
    valid_text = (
        "Item 1. Business. Apple Inc. designs, manufactures, and markets smartphones, "
        "personal computers, tablets, wearables, and accessories, and sells a variety of "
        "related services. The Company's fiscal year is the 52 or 53-week period that ends "
        "on the last Saturday of September. The Company is committed to bringing the best "
        "user experience to customers through innovative hardware, software, and services."
    )
    assert is_informative_chunk(valid_text)


def test_prepare_training_data():
    sample_data = {
        "text": [
            "Apple Inc. produces iPhone in California. Tim Cook leads Apple Inc.",
            "Microsoft Corporation acquired LinkedIn for $26 billion."
        ],
        "triples": [
            [
                {
                    "head": "Apple Inc.",
                    "head_type": "ORG",
                    "relation": "Produces",
                    "tail": "iPhone",
                    "tail_type": "PRODUCT"
                },
                {
                    "head": "Apple Inc.",
                    "head_type": "ORG",
                    "relation": "Led_By",
                    "tail": "Tim Cook",
                    "tail_type": "PERSON"
                }
            ],
            [
                {
                    "head": "Microsoft Corporation",
                    "head_type": "ORG",
                    "relation": "Operates_In",
                    "tail": "LinkedIn",
                    "tail_type": "ORG"
                }
            ]
        ]
    }
    df = pd.DataFrame(sample_data)

    params = {
        "encoder_model_name": "bert-base-uncased",
        "max_seq_length": 64,
        "max_gt_triples": 5
    }

    dataset, tokenizer = prepare_training_data(df, params)

    assert "input_ids" in dataset
    assert "attention_mask" in dataset
    assert "relations" in dataset
    assert "subj_types" in dataset
    assert "obj_types" in dataset
    assert "subj_spans" in dataset
    assert "obj_spans" in dataset

    assert len(dataset["input_ids"]) >= 1
    assert dataset["relations"].shape[1] == 5
    assert dataset["subj_spans"].shape[1] == 5

    # Test namespaced call signature
    data_prep_p = {"max_seq_length": 64, "max_gt_triples": 3, "max_samples": 10}
    training_p = {"encoder_model_name": "bert-base-uncased"}
    schema_p = {"entity_types": ["org", "product", "person"], "relation_types": ["produces", "led_by"]}
    ds2, tok2 = prepare_training_data(df, data_prep_p, training_p, schema_p)
    assert len(ds2["input_ids"]) >= 1
    assert ds2["relations"].shape[1] == 3


def test_parse_sec_filings_missing_docling():
    with patch("bert_kg_mvp.pipelines.data_prep.nodes.DocumentConverter", None):
        with pytest.raises(ImportError, match="docling is required"):
            parse_sec_filings("/some/dir")


def test_parse_sec_filings_mocked():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a mock PDF
        pdf_path = os.path.join(tmp_dir, "SAMPLE_2024.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 dummy")

        item1 = MagicMock()
        item1.text = "Item 1. Business Overview"
        item2 = MagicMock()
        item2.text = ("Apple designs iPhones and personal computers across multiple regions. " * 15)
        item3 = MagicMock()
        item3.text = "Item 9. Controls and Procedures"

        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = [
            (item1, 0),
            (item2, 1),
            (item3, 0),
        ]

        mock_conv_result = MagicMock()
        mock_conv_result.document = mock_doc
        mock_converter_instance = MagicMock()
        mock_converter_instance.convert.return_value = mock_conv_result

        MockConverterClass = MagicMock(return_value=mock_converter_instance)

        with patch("bert_kg_mvp.pipelines.data_prep.nodes.DocumentConverter", MockConverterClass):
            # Positional string argument
            df = parse_sec_filings(tmp_dir, max_words=50)
            assert isinstance(df, pd.DataFrame)
            assert len(df) >= 1

            # Dict argument (namespaced params)
            df_dict = parse_sec_filings({"raw_pdf_dir": tmp_dir, "max_chunk_words": 50})
            assert isinstance(df_dict, pd.DataFrame)
            assert len(df_dict) >= 1


def test_data_prep_pipeline_structure():
    pipeline = create_pipeline()
    node_names = [n.name for n in pipeline.nodes]
    assert "parse_sec_filings_node" in node_names
    assert "generate_teacher_triplets_node" in node_names


def test_extract_rebel_triplets():
    rebel_str = "<s> <triplet> Apple <subj> iPhone <obj> produces </s>"
    triplets = extract_rebel_triplets(rebel_str)
    assert len(triplets) == 1
    assert triplets[0]["head"] == "Apple"
    assert triplets[0]["tail"] == "iPhone"
    assert triplets[0]["type"] == "produces"

