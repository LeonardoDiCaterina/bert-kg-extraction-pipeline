import torch
import torch.nn as nn
from torch.optim import AdamW
from bert_kg_mvp.models.architecture import BERTToKnowledgeGraph

def train_model(processed_data: dict, tokenizer, parameters: dict):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Training on device: {device}")
    
    # Initialize model with dynamically sized vocabulary
    vocab_size = len(tokenizer)
    model = BERTToKnowledgeGraph(vocab_size=vocab_size, d_model=parameters["d_model"])
    model.to(device)
    
    # Setup optimizer and loss function
    # Note: CrossEntropyLoss inherently ignores targets labeled as -100
    optimizer = AdamW(model.parameters(), lr=parameters["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    # Load tensors to device
    input_ids = processed_data["input_ids"].to(device)
    attention_mask = processed_data["attention_mask"].to(device)
    decoder_input_ids = processed_data["decoder_input_ids"].to(device)
    labels = processed_data["labels"].to(device)
    
    epochs = parameters.get("epochs", 3)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass (Teacher Forcing)
        logits = model(input_ids, attention_mask, decoder_input_ids)
        
        # Flatten logits and labels for loss calculation
        # Logits shape: (Batch * Seq_Len, Vocab_Size)
        # Labels shape: (Batch * Seq_Len)
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
        
    return model.cpu()
