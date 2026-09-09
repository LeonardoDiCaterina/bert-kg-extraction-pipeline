import gc
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from bert_kg_mvp.models.architecture_2 import BERTToKnowledgeGraph_2

def train_model(processed_dataset: dict, tokenizer, parameters: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    vocab_size = len(tokenizer)
    model = BERTToKnowledgeGraph_2(vocab_size=vocab_size).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters.get("learning_rate", 5e-5))
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    dataset = TensorDataset(
        processed_dataset["input_ids"],
        processed_dataset["attention_mask"],
        processed_dataset["decoder_input_ids"],
        processed_dataset["labels"]
    )
    
    batch_size = parameters.get("batch_size", 4)
    accum_steps = parameters.get("gradient_accumulation_steps", 8)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    # FIX: Force the frozen encoder into eval mode to disable dropout and bypass the MPS SDPA bug
    model.encoder.eval()
    
    epochs = parameters.get("epochs", 10)
    
    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(dataloader):
            b_input_ids, b_attn_mask, b_dec_ids, b_labels = [b.to(device) for b in batch]
            
            logits = model(b_input_ids, b_attn_mask, b_dec_ids)
            loss = criterion(logits.view(-1, vocab_size), b_labels.view(-1))
            
            (loss / accum_steps).backward()
            
            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if step % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {step}/{len(dataloader)} | Loss: {loss.item():.4f}")
            
            if str(device) == "mps": torch.mps.empty_cache()
            del logits, loss, b_input_ids, b_attn_mask, b_dec_ids, b_labels
            if step % 50 == 0: gc.collect()
            
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {(total_loss / len(dataloader)):.4f}")
        
    return model
