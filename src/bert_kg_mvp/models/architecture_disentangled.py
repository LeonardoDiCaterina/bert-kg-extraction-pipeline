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


class DynamicKGExtractorDisentangled(nn.Module):
    def __init__(
        self,
        encoder_model_name="bert-base-uncased",
        d_model=768,
        span_d_model=256,
        num_layers=4,
        num_queries=15,
        num_relations=5,
        num_ent_types=7,
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
            for param in self.encoder.parameters():
                param.requires_grad = False

            encoder_layers = None
            if hasattr(self.encoder, "encoder") and hasattr(
                self.encoder.encoder, "layer"
            ):
                encoder_layers = self.encoder.encoder.layer
            elif hasattr(self.encoder, "layer"):
                encoder_layers = self.encoder.layer

            if encoder_layers is not None:
                for layer in encoder_layers[-unfrozen_top_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            else:
                print(
                    "WARNING: Could not automatically detect encoder layers for partial freezing. Falling back to freezing all."
                )

        self.num_queries = num_queries

        # Learned object queries (shared between branches)
        self.query_embed = nn.Embedding(num_queries, d_model)

        # ----------------------------------------------------
        # BRANCH 1: Semantic Decoder (Type + Relation)
        # ----------------------------------------------------
        sem_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=8, batch_first=True
        )
        self.sem_decoder = nn.TransformerDecoder(sem_decoder_layer, num_layers=num_layers)
        
        self.dropout_sem = nn.Dropout(0.3)
        self.rel_class_head = nn.Linear(d_model, num_relations + 1)
        self.subj_type_head = nn.Linear(d_model, num_ent_types)
        self.obj_type_head = nn.Linear(d_model, num_ent_types)

        # ----------------------------------------------------
        # BRANCH 2: Span Decoder (Pointer Networks)
        # ----------------------------------------------------
        self.span_memory_proj = nn.Linear(d_model, span_d_model)
        self.span_query_proj = nn.Linear(d_model, span_d_model)
        
        span_decoder_layer = nn.TransformerDecoderLayer(
            d_model=span_d_model, nhead=4, batch_first=True
        )
        self.span_decoder = nn.TransformerDecoder(span_decoder_layer, num_layers=num_layers)
        
        self.dropout_span = nn.Dropout(0.3)
        
        # Pointer networks project from span_d_model back to d_model for the dot product with encoder memory
        self.subj_start_ptr = nn.Linear(span_d_model, d_model)
        self.subj_end_ptr = nn.Linear(span_d_model, d_model)
        self.obj_start_ptr = nn.Linear(span_d_model, d_model)
        self.obj_end_ptr = nn.Linear(span_d_model, d_model)

    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        memory = encoder_outputs.last_hidden_state
        bs = input_ids.size(0)

        # Shared query embeddings
        query_embeds = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        # ----------------------------------------------------
        # Semantic Branch Pass
        # ----------------------------------------------------
        sem_decoder_out = self.sem_decoder(query_embeds, memory)
        sem_decoder_out = self.dropout_sem(sem_decoder_out)
        
        rel_logits = self.rel_class_head(sem_decoder_out)
        subj_type_logits = self.subj_type_head(sem_decoder_out)
        obj_type_logits = self.obj_type_head(sem_decoder_out)

        # ----------------------------------------------------
        # Span Branch Pass
        # ----------------------------------------------------
        span_memory = self.span_memory_proj(memory)
        span_query_embeds = self.span_query_proj(query_embeds)
        
        span_decoder_out = self.span_decoder(span_query_embeds, span_memory)
        span_decoder_out = self.dropout_span(span_decoder_out)

        # Dot product with ORIGINAL memory (not projected memory) to retain rich positional features
        mem_t = memory.transpose(1, 2)

        subj_start_logits = torch.bmm(self.subj_start_ptr(span_decoder_out), mem_t)
        subj_end_logits = torch.bmm(self.subj_end_ptr(span_decoder_out), mem_t)
        obj_start_logits = torch.bmm(self.obj_start_ptr(span_decoder_out), mem_t)
        obj_end_logits = torch.bmm(self.obj_end_ptr(span_decoder_out), mem_t)

        return {
            "rel_logits": rel_logits,
            "subj_type_logits": subj_type_logits,
            "obj_type_logits": obj_type_logits,
            "subj_start_logits": subj_start_logits,
            "subj_end_logits": subj_end_logits,
            "obj_start_logits": obj_start_logits,
            "obj_end_logits": obj_end_logits,
        }
