import gc
import torch
from torch.utils.data import TensorDataset, DataLoader
from bert_kg_mvp.models.bipartite_loss import SetCriterion
from bert_kg_mvp.models.architecture_2 import DynamicKGExtractor

def train_model(processed_dataset: dict, tokenizer, parameters: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    # Extract dynamic architecture parameters
    encoder_model_name = parameters.get("encoder_model_name", "bert-base-uncased")
    decoder_num_layers = parameters.get("decoder_num_layers", 4)
    num_queries = parameters.get("num_queries", 15)
    freeze_strategy = parameters.get("freeze_strategy", "partial")
    unfrozen_top_layers = parameters.get("unfrozen_top_layers", 4)
    d_model = parameters.get("d_model", 768)
    
    # Instantiate the new dynamic architecture
    model = DynamicKGExtractor(
        encoder_model_name=encoder_model_name,
        d_model=d_model,
        num_layers=decoder_num_layers,
        num_queries=num_queries,
        num_relations=5,
        num_ent_types=7,
        freeze_strategy=freeze_strategy,
        unfrozen_top_layers=unfrozen_top_layers
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters.get("learning_rate", 5e-5))
    
    # Bipartite Matching Loss
    criterion = SetCriterion(num_relation_classes=5, num_entity_types=7).to(device)
    
    dataset = TensorDataset(
        processed_dataset["input_ids"],
        processed_dataset["attention_mask"],
        processed_dataset["relations"],
        processed_dataset["subj_types"],
        processed_dataset["obj_types"],
        processed_dataset["subj_spans"],
        processed_dataset["obj_spans"]
    )
    
    batch_size = parameters.get("batch_size", 4)
    accum_steps = parameters.get("gradient_accumulation_steps", 8)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    if freeze_strategy == "all":
        # FIX: Force the frozen encoder into eval mode to disable dropout and bypass the MPS SDPA bug
        model.encoder.eval()
    elif freeze_strategy == "partial":
        # Force bottom layers into eval mode, leave top layers in train mode
        model.encoder.eval()
        # Find layers to re-enable train mode
        encoder_layers = None
        if hasattr(model.encoder, "encoder") and hasattr(model.encoder.encoder, "layer"):
            encoder_layers = model.encoder.encoder.layer
        elif hasattr(model.encoder, "layer"):
            encoder_layers = model.encoder.layer
            
        if encoder_layers is not None:
            for layer in encoder_layers[-unfrozen_top_layers:]:
                layer.train()
    
    epochs = parameters.get("epochs", 10)
    
    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(dataloader):
            batch = [b.to(device) for b in batch]
            b_input_ids, b_attn_mask, b_relations, b_subj_types, b_obj_types, b_subj_spans, b_obj_spans = batch
            
            # Forward pass (no decoder_input_ids)
            outputs = model(b_input_ids, b_attn_mask)
            
            # Prepare targets for the Hungarian matcher
            targets = []
            for i in range(b_input_ids.size(0)):
                # Filter out padded dummy 'no_relation' (idx=5)
                valid_idx = b_relations[i] != 5
                targets.append({
                    "relations": b_relations[i][valid_idx],
                    "subj_types": b_subj_types[i][valid_idx],
                    "obj_types": b_obj_types[i][valid_idx],
                    "subj_spans": b_subj_spans[i][valid_idx],
                    "obj_spans": b_obj_spans[i][valid_idx]
                })
                
            # Compute bipartite matching loss
            loss_dict = criterion(outputs, targets)
            
            # Combine all parts of the loss
            loss = sum(loss_dict.values())
            
            (loss / accum_steps).backward()
            
            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if step % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {step}/{len(dataloader)} | Loss: {loss.item():.4f}")
            
            if str(device) == "mps":
                torch.mps.empty_cache()
            del outputs, loss, loss_dict, batch
            if step % 50 == 0:
                gc.collect()
            
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {(total_loss / len(dataloader)):.4f}")
        
    return model
