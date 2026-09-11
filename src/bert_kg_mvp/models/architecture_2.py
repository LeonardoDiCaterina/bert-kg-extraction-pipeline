import math
import torch
import torch.nn as nn
from transformers import AutoModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class DynamicKGExtractor(nn.Module):
    def __init__(
        self, 
        encoder_model_name="bert-base-uncased", 
        d_model=768, 
        num_layers=4, 
        num_queries=15, 
        num_relations=5, 
        num_ent_types=7,
        freeze_strategy="partial",
        unfrozen_top_layers=4
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
            if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
                encoder_layers = self.encoder.encoder.layer # BERT, RoBERTa
            elif hasattr(self.encoder, "layer"):
                encoder_layers = self.encoder.layer # Some older architectures
                
            if encoder_layers is not None:
                # Unfreeze the top N layers
                for layer in encoder_layers[-unfrozen_top_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            else:
                print("WARNING: Could not automatically detect encoder layers for partial freezing. Falling back to freezing all.")
            
        self.num_queries = num_queries
        
        # Learned object queries instead of target embeddings
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Prediction Heads
        self.rel_class_head = nn.Linear(d_model, num_relations + 1)
        self.subj_type_head = nn.Linear(d_model, num_ent_types)
        self.obj_type_head = nn.Linear(d_model, num_ent_types)
        
        # Pointer Networks for Entity Spans
        self.subj_start_ptr = nn.Linear(d_model, d_model)
        self.subj_end_ptr = nn.Linear(d_model, d_model)
        self.obj_start_ptr = nn.Linear(d_model, d_model)
        self.obj_end_ptr = nn.Linear(d_model, d_model)

    def forward(self, input_ids, attention_mask):
        # We don't use torch.no_grad() here because we may have unfrozen top layers
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        memory = encoder_outputs.last_hidden_state

        bs = input_ids.size(0)
        
        # Expand learned queries for the batch
        query_embeds = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        
        # No causal mask needed for set prediction (bidirectional cross-attention)
        decoder_output = self.decoder(query_embeds, memory)
        
        # Classification logits
        rel_logits = self.rel_class_head(decoder_output)
        subj_type_logits = self.subj_type_head(decoder_output)
        obj_type_logits = self.obj_type_head(decoder_output)
        
        # Pointer Network Logits (dot product with encoder memory)
        # memory shape: [B, seq_len, d_model] -> transposed: [B, d_model, seq_len]
        # query shape: [B, num_queries, d_model]
        mem_t = memory.transpose(1, 2)
        
        subj_start_logits = torch.bmm(self.subj_start_ptr(decoder_output), mem_t) # [B, num_queries, seq_len]
        subj_end_logits = torch.bmm(self.subj_end_ptr(decoder_output), mem_t)
        obj_start_logits = torch.bmm(self.obj_start_ptr(decoder_output), mem_t)
        obj_end_logits = torch.bmm(self.obj_end_ptr(decoder_output), mem_t)
        
        return {
            "rel_logits": rel_logits,
            "subj_type_logits": subj_type_logits,
            "obj_type_logits": obj_type_logits,
            "subj_start_logits": subj_start_logits,
            "subj_end_logits": subj_end_logits,
            "obj_start_logits": obj_start_logits,
            "obj_end_logits": obj_end_logits
        }
