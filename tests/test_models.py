import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from bert_kg_mvp.models.architecture import (
    PositionalEncoding as PosEnc1,
    BERTToKnowledgeGraph,
)
from bert_kg_mvp.models.architecture_2 import (
    PositionalEncoding as PosEnc2,
    DynamicKGExtractor,
)
from bert_kg_mvp.models.logits_processor import KGSchemaLogitsProcessor


def test_positional_encoding():
    x = torch.zeros(2, 32, 64)

    pe1 = PosEnc1(d_model=64, max_len=128)
    out1 = pe1(x)
    assert out1.shape == (2, 32, 64)
    assert not torch.all(out1 == 0)

    pe2 = PosEnc2(d_model=64, max_len=128)
    out2 = pe2(x)
    assert out2.shape == (2, 32, 64)
    assert not torch.all(out2 == 0)


def test_kg_schema_logits_processor():
    mock_tokenizer = MagicMock()
    mock_tokenizer.convert_tokens_to_ids.side_effect = lambda t: {
        "|": 10,
        "[EOT]": 11,
        "[EOS]": 12,
        "ORG": 20,
        "LOC": 21,
    }.get(t, 99)
    mock_tokenizer.encode.side_effect = lambda text, add_special_tokens=False: {
        "ACQUIRED": [30],
        "SUBSIDIARY_OF": [31],
    }.get(text, [99])

    processor = KGSchemaLogitsProcessor(
        tokenizer=mock_tokenizer,
        valid_types=["ORG", "LOC"],
        valid_relations=["ACQUIRED", "SUBSIDIARY_OF"],
    )

    # Batch of size 3 with different pipe counts:
    # seq 0: no pipe -> expects pipe_id (10)
    # seq 1: 1 pipe and 0 tokens since pipe -> expects type (20, 21)
    # seq 2: 1 pipe and >0 tokens since pipe -> expects second pipe (10)
    # seq 3: 2 pipes -> expects relation (30, 31) or EOT (11) / EOS (12)
    # 1. No pipe -> expects pipe_id (10)
    out0 = processor(torch.tensor([[1, 2, 3]]), torch.zeros(1, 50))
    assert out0[0, 10] == 0.0
    assert out0[0, 0] == float("-inf")

    # 2. One pipe, 0 tokens since pipe -> expects entity type (20, 21)
    out1 = processor(torch.tensor([[1, 10]]), torch.zeros(1, 50))
    assert out1[0, 20] == 0.0
    assert out1[0, 21] == 0.0
    assert out1[0, 10] == float("-inf")

    # 3. One pipe, >0 tokens since pipe -> expects closing pipe (10)
    out2 = processor(torch.tensor([[1, 10, 20]]), torch.zeros(1, 50))
    assert out2[0, 10] == 0.0
    assert out2[0, 20] == float("-inf")

    # 4. Two pipes -> expects relation (30, 31) or EOT (11) / EOS (12)
    out3 = processor(torch.tensor([[1, 10, 20, 10]]), torch.zeros(1, 50))
    assert out3[0, 30] == 0.0
    assert out3[0, 31] == 0.0
    assert out3[0, 11] == 0.0
    assert out3[0, 12] == 0.0
    assert out3[0, 0] == float("-inf")

    # 5. Greater than 2 pipes -> continue / untouched
    out4 = processor(torch.tensor([[1, 10, 20, 10, 30, 10]]), torch.zeros(1, 50))
    assert out4[0, 0] == 0.0


class DummyEncoderOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class DummyEncoder(nn.Module):
    def __init__(self, d_model=64, num_layers=4):
        super().__init__()
        self.d_model = d_model
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_layers)]
        )
        self.embedding = nn.Embedding(100, d_model)

    def forward(self, input_ids, attention_mask=None):
        bs, seq_len = input_ids.shape
        x = self.embedding(input_ids)
        for layer in self.encoder.layer:
            x = layer(x)
        return DummyEncoderOutput(x)


@patch("transformers.AutoModel.from_pretrained")
def test_dynamic_kg_extractor_freezing_strategies(mock_from_pretrained):
    dummy_enc = DummyEncoder(d_model=64, num_layers=4)
    mock_from_pretrained.return_value = dummy_enc

    # Strategy: "all"
    model_all = DynamicKGExtractor(d_model=64, freeze_strategy="all", num_queries=5)
    for p in model_all.encoder.parameters():
        assert not p.requires_grad

    # Strategy: "none"
    dummy_enc_none = DummyEncoder(d_model=64, num_layers=4)
    mock_from_pretrained.return_value = dummy_enc_none
    model_none = DynamicKGExtractor(d_model=64, freeze_strategy="none", num_queries=5)
    for p in model_none.encoder.parameters():
        assert p.requires_grad

    # Strategy: "partial"
    dummy_enc_partial = DummyEncoder(d_model=64, num_layers=4)
    mock_from_pretrained.return_value = dummy_enc_partial
    model_partial = DynamicKGExtractor(
        d_model=64, freeze_strategy="partial", unfrozen_top_layers=2, num_queries=5
    )
    # First 2 layers frozen, last 2 unfrozen
    for p in model_partial.encoder.encoder.layer[0].parameters():
        assert not p.requires_grad
    for p in model_partial.encoder.encoder.layer[-1].parameters():
        assert p.requires_grad

    # Fallback when encoder layers not detected
    dummy_enc_no_layers = nn.Linear(64, 64)
    mock_from_pretrained.return_value = dummy_enc_no_layers
    model_fallback = DynamicKGExtractor(
        d_model=64, freeze_strategy="partial", num_queries=5
    )
    for p in model_fallback.encoder.parameters():
        assert not p.requires_grad


@patch("transformers.AutoModel.from_pretrained")
def test_dynamic_kg_extractor_forward(mock_from_pretrained):
    dummy_enc = DummyEncoder(d_model=64, num_layers=2)
    mock_from_pretrained.return_value = dummy_enc

    model = DynamicKGExtractor(
        d_model=64, num_layers=2, num_queries=6, num_relations=4, num_ent_types=5
    )

    bs, seq_len = 2, 10
    input_ids = torch.randint(0, 100, (bs, seq_len))
    attention_mask = torch.ones((bs, seq_len))

    outputs = model(input_ids, attention_mask)

    assert outputs["rel_logits"].shape == (bs, 6, 5)  # num_relations + 1
    assert outputs["subj_type_logits"].shape == (bs, 6, 5)
    assert outputs["obj_type_logits"].shape == (bs, 6, 5)
    assert outputs["subj_start_logits"].shape == (bs, 6, seq_len)
    assert outputs["subj_end_logits"].shape == (bs, 6, seq_len)
    assert outputs["obj_start_logits"].shape == (bs, 6, seq_len)
    assert outputs["obj_end_logits"].shape == (bs, 6, seq_len)


@patch("transformers.BertModel.from_pretrained")
def test_bert_to_knowledge_graph(mock_from_pretrained):
    dummy_enc = DummyEncoder(d_model=64, num_layers=2)
    mock_from_pretrained.return_value = dummy_enc

    model = BERTToKnowledgeGraph(vocab_size=120, d_model=64, num_layers=2)

    bs, seq_len, dec_len = 2, 8, 5
    input_ids = torch.randint(0, 100, (bs, seq_len))
    attention_mask = torch.ones((bs, seq_len))
    dec_input_ids = torch.randint(0, 100, (bs, dec_len))

    logits = model(input_ids, attention_mask, dec_input_ids)
    assert logits.shape == (bs, dec_len, 120)
