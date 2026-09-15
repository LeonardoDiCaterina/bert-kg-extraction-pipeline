import math
import torch
import torch.nn as nn
from transformers import AutoModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


class DynamicKGExtractorTypedQueries(nn.Module):
    def __init__(
        self,
        encoder_model_name="bert-base-uncased",
        d_model=768,
        num_layers=4,
        queries_per_rel=3,
        num_relations=5,
        num_ent_types=7,
        num_token_slots=8,
        freeze_strategy="partial",
        unfrozen_top_layers=4,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model_name)

        # 🧊 Dynamic Freezing Strategy
        if freeze_strategy == "all":
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif freeze_strategy == "partial":
            # Freeze everything first
            for param in self.encoder.parameters():
                param.requires_grad = False

            # Attempt to find the transformer layers dynamically
            encoder_layers = None
            if hasattr(self.encoder, "encoder") and hasattr(
                self.encoder.encoder, "layer"
            ):
                encoder_layers = self.encoder.encoder.layer  # BERT, RoBERTa
            elif hasattr(self.encoder, "layer"):
                encoder_layers = self.encoder.layer  # Some older architectures

            if encoder_layers is not None:
                # Unfreeze the top N layers
                for layer in encoder_layers[-unfrozen_top_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            else:
                print(
                    "WARNING: Could not automatically detect encoder layers for partial freezing. Falling back to freezing all."
                )

        self.queries_per_rel = queries_per_rel
        self.num_relations = num_relations
        self.num_queries = queries_per_rel * num_relations

        # Learned object queries instead of target embeddings
        self.query_embed = nn.Embedding(self.num_queries, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=8, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Prediction Heads
        self.dropout = nn.Dropout(0.3)
        self.rel_class_head = nn.Linear(d_model, num_relations + 1)
        self.subj_type_head = nn.Linear(d_model, num_ent_types)
        self.obj_type_head = nn.Linear(d_model, num_ent_types)

        # Multi-Token Slot Networks for Entity Spans (independent heads for subject and object)
        self.num_token_slots = num_token_slots
        self.subj_ptr_proj = nn.Linear(d_model, d_model)
        self.obj_ptr_proj = nn.Linear(d_model, d_model)
        self.subj_slot_embed = nn.Embedding(num_token_slots, d_model)
        self.obj_slot_embed = nn.Embedding(num_token_slots, d_model)
        self.subj_null_bias = nn.Parameter(torch.zeros(num_token_slots))
        self.obj_null_bias = nn.Parameter(torch.zeros(num_token_slots))

    def forward(self, input_ids, attention_mask):
        # We don't use torch.no_grad() here because we may have unfrozen top layers
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        memory = encoder_outputs.last_hidden_state

        bs = input_ids.size(0)

        # Expand learned queries for the batch
        query_embeds = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        # No causal mask needed for set prediction (bidirectional cross-attention)
        decoder_output = self.decoder(query_embeds, memory)

        # Apply dropout to combat dataset memorization
        decoder_output = self.dropout(decoder_output)

        # Classification logits
        rel_logits = self.rel_class_head(decoder_output)
        subj_type_logits = self.subj_type_head(decoder_output)
        obj_type_logits = self.obj_type_head(decoder_output)

        # Multi-Token Slot Logits (dot product with encoder memory + null bias)
        mem_t = memory.transpose(1, 2)  # [B, d_model, seq_len]

        subj_slot_logits = []
        obj_slot_logits = []
        for k in range(self.num_token_slots):
            # Subject
            s_slot_offset = self.subj_slot_embed.weight[k]  # [d_model]
            subj_proj = self.subj_ptr_proj(decoder_output + s_slot_offset)  # [B, Q, d_model]
            subj_token_logits = torch.bmm(subj_proj, mem_t)                 # [B, Q, seq_len]
            subj_null = self.subj_null_bias[k].expand(bs, self.num_queries, 1)
            subj_slot_logits.append(torch.cat([subj_token_logits, subj_null], dim=-1))
            
            # Object
            o_slot_offset = self.obj_slot_embed.weight[k]  # [d_model]
            obj_proj = self.obj_ptr_proj(decoder_output + o_slot_offset)    # [B, Q, d_model]
            obj_token_logits = torch.bmm(obj_proj, mem_t)                  # [B, Q, seq_len]
            obj_null = self.obj_null_bias[k].expand(bs, self.num_queries, 1)
            obj_slot_logits.append(torch.cat([obj_token_logits, obj_null], dim=-1))

        subj_slot_logits = torch.stack(subj_slot_logits, dim=2)  # [B, Q, K, seq_len+1]
        obj_slot_logits = torch.stack(obj_slot_logits, dim=2)

        return {
            "rel_logits": rel_logits,
            "subj_type_logits": subj_type_logits,
            "obj_type_logits": obj_type_logits,
            "subj_slot_logits": subj_slot_logits,
            "obj_slot_logits": obj_slot_logits,
        }
