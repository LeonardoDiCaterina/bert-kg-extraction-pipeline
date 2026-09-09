import math
import torch
import torch.nn as nn
from transformers import BertModel

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

class BERTToKnowledgeGraph_2(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=4):
        super().__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        
        # 🧊 FREEZE THE ENCODER: Drastically speeds up training
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.token_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        # Disable gradient tracking for the encoder forward pass
        with torch.no_grad():
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        memory = encoder_outputs.last_hidden_state

        tgt_emb = self.embedding(decoder_input_ids)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        seq_len = decoder_input_ids.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(decoder_input_ids.device)
        
        decoder_output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        logits = self.token_head(decoder_output)
        
        return logits
