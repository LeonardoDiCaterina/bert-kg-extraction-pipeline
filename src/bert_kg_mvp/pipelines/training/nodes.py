import gc
from typing import Any, Dict, Optional
import torch
from torch.utils.data import TensorDataset, DataLoader
from bert_kg_mvp.models.bipartite_loss import SetCriterion
from bert_kg_mvp.models import build_decoder


def train_model(
    processed_dataset: Dict[str, torch.Tensor],
    tokenizer: Any,
    training_params: Dict[str, Any],
    schema_params: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Trains the DynamicKGExtractor model using DETR-style bipartite matching loss.
    Supports either namespaced params (`params:training` + `params:schema`) or a legacy single dict.
    """
    matmul_precision = training_params.get("float32_matmul_precision", "high")
    if matmul_precision and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    if schema_params is None:
        schema_params = training_params.get("schema", {})

    relation_types = schema_params.get(
        "relation_types",
        ["has_metric", "produces", "operates_in", "reports_risk", "led_by"],
    )
    entity_types = schema_params.get(
        "entity_types",
        ["org", "person", "product", "segment", "fin_metric", "risk_factor", "event"],
    )

    num_relations = len(relation_types)
    num_ent_types = len(entity_types)
    no_relation_idx = num_relations

    # Extract dynamic architecture parameters
    encoder_model_name = training_params.get("encoder_model_name", "bert-base-uncased")
    decoder_num_layers = training_params.get("decoder_num_layers", 4)
    num_queries = training_params.get("num_queries", 15)
    freeze_strategy = training_params.get("freeze_strategy", "partial")
    unfrozen_top_layers = training_params.get("unfrozen_top_layers", 4)
    d_model = training_params.get("d_model", 768)

    decoder_type = training_params.get("decoder_type", "baseline")
    queries_per_rel = training_params.get("queries_per_rel", None)
    span_d_model = training_params.get("span_d_model", 256)

    # Base kwargs for all decoders
    kwargs = {
        "encoder_model_name": encoder_model_name,
        "d_model": d_model,
        "num_layers": decoder_num_layers,
        "num_relations": num_relations,
        "num_ent_types": num_ent_types,
        "freeze_strategy": freeze_strategy,
        "unfrozen_top_layers": unfrozen_top_layers,
    }

    # Add specific kwargs based on decoder type
    if decoder_type == "baseline":
        kwargs["num_queries"] = num_queries
    elif decoder_type == "typed":
        kwargs["queries_per_rel"] = queries_per_rel if queries_per_rel is not None else num_queries // num_relations
    elif decoder_type == "disentangled":
        kwargs["num_queries"] = num_queries
        kwargs["span_d_model"] = span_d_model

    model = build_decoder(decoder_type, **kwargs).to(device)

    learning_rate = training_params.get("learning_rate", 1e-4)
    encoder_lr = training_params.get("encoder_learning_rate", 1e-5)

    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    weight_decay = training_params.get("weight_decay", 0.05)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": decoder_params, "lr": learning_rate},
        ],
        weight_decay=weight_decay
    )
    criterion = SetCriterion(
        num_relation_classes=num_relations, 
        num_entity_types=num_ent_types,
        queries_per_rel=queries_per_rel if decoder_type == "typed" else None
    ).to(device)

    from bert_kg_mvp.utils.dataset import AugmentedKGDataset
    prefix_end_token_id = tokenizer.convert_tokens_to_ids("]") if "]" in tokenizer.get_vocab() else None
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else pad_token_id
    
    dataset = AugmentedKGDataset(
        tensors_dict=processed_dataset,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        no_relation_idx=no_relation_idx,
        prefix_end_token_id=prefix_end_token_id,
        mask_prob=training_params.get("mask_prob", 0.15),
        prefix_drop_prob=training_params.get("prefix_drop_prob", 0.15),
        span_jitter_prob=training_params.get("span_jitter_prob", 0.1)
    )

    batch_size = training_params.get("batch_size", 4)
    accum_steps = training_params.get("gradient_accumulation_steps", 8)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    if freeze_strategy == "all":
        model.encoder.eval()
    elif freeze_strategy == "partial":
        model.encoder.eval()
        encoder_layers = None
        if hasattr(model.encoder, "encoder") and hasattr(
            model.encoder.encoder, "layer"
        ):
            encoder_layers = model.encoder.encoder.layer
        elif hasattr(model.encoder, "layer"):
            encoder_layers = model.encoder.layer

        if encoder_layers is not None:
            for layer in encoder_layers[-unfrozen_top_layers:]:
                layer.train()

    # Optional torch.compile acceleration (fuses multi-head attention kernels & heads)
    compile_model = training_params.get("compile_model", False)
    if compile_model:
        if hasattr(torch, "compile"):
            compile_mode = training_params.get("compile_mode", "default")
            try:
                print(
                    f"Compiling DynamicKGExtractor with torch.compile(mode='{compile_mode}')..."
                )
                model = torch.compile(model, mode=compile_mode)
            except Exception as e:
                print(f"Warning: torch.compile failed ({e}). Proceeding uncompiled.")
        else:
            print(
                "torch.compile is not available in this PyTorch version. Proceeding uncompiled."
            )

    epochs = training_params.get("epochs", 10)

    from collections import defaultdict
    import pandas as pd
    import os
    
    history = []
    
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    ema_decay = training_params.get("ema_decay", 0.999)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))

    checkpoint_config = training_params.get("checkpointing", {})
    use_checkpointing = checkpoint_config.get("enabled", False)
    
    if use_checkpointing:
        from bert_kg_mvp.utils.checkpointer import ModelCheckpointer
        checkpointer = ModelCheckpointer(
            checkpoint_dir=checkpoint_config.get("dirpath", "data/06_models/checkpoints"),
            model_name="production_model",
            mode="min",  # Minimizing training loss
            save_top_k=checkpoint_config.get("save_top_k", 1),
            save_last=checkpoint_config.get("save_last", True),
        )
    else:
        checkpointer = None

    for epoch in range(epochs):
        total_loss = 0.0
        epoch_losses = defaultdict(float)
        optimizer.zero_grad()
        for step, batch in enumerate(dataloader):
            batch = [b.to(device) for b in batch]
            (
                b_input_ids,
                b_attn_mask,
                b_relations,
                b_subj_types,
                b_obj_types,
                b_subj_spans,
                b_obj_spans,
            ) = batch

            outputs = model(b_input_ids, b_attn_mask)

            targets = []
            for i in range(b_input_ids.size(0)):
                valid_idx = b_relations[i] != no_relation_idx
                targets.append(
                    {
                        "relations": b_relations[i][valid_idx],
                        "subj_types": b_subj_types[i][valid_idx],
                        "obj_types": b_obj_types[i][valid_idx],
                        "subj_spans": b_subj_spans[i][valid_idx],
                        "obj_spans": b_obj_spans[i][valid_idx],
                    }
                )

            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())
            (loss / accum_steps).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                ema_model.update_parameters(model)
                optimizer.zero_grad()

            total_loss += loss.item()
            for k, v in loss_dict.items():
                epoch_losses[k] += v.item()

            if step % 100 == 0:
                components_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
                print(
                    f"Epoch {epoch + 1}/{epochs} | Batch {step}/{len(dataloader)} | Total: {loss.item():.4f} | {components_str}"
                )

            if str(device) == "mps":
                torch.mps.empty_cache()
            del outputs, loss, loss_dict, batch
            if step % 50 == 0:
                gc.collect()

        avg_total = total_loss / len(dataloader)
        avg_losses = {k: v / len(dataloader) for k, v in epoch_losses.items()}
        
        # Calculate L2 Norm of model parameters to show weight decay regularization effect
        l2_norm = sum(p.norm(2).item() for p in model.parameters() if p.requires_grad)
        
        print(f"Epoch {epoch + 1}/{epochs} - Avg Total Loss: {avg_total:.4f}")
        avg_components = " | ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
        print(f"  --> Components: {avg_components} | l2_norm: {l2_norm:.2f}")
        
        # Track history
        history_record = {"epoch": epoch + 1, "total_loss": avg_total}
        history_record.update(avg_losses)
        history.append(history_record)
        
        if use_checkpointing:
            checkpointer.save_checkpoint(
                epoch=epoch + 1,
                model=ema_model,
                optimizer=optimizer,
                scheduler=None,
                metric_value=avg_total
            )

    # Save tracking history to CSV
    os.makedirs("data/08_reporting", exist_ok=True)
    safe_model_name = encoder_model_name.replace("/", "_")
    history_df = pd.DataFrame(history)
    history_df.to_csv(f"data/08_reporting/loss_history_{safe_model_name}.csv", index=False)
    print(f"Saved loss history to data/08_reporting/loss_history_{safe_model_name}.csv")

    # Load EMA weights into the raw model before returning
    model.load_state_dict(ema_model.module.state_dict())
    
    # Return underlying uncompiled model if wrapped (for clean serialization)
    return getattr(model, "_orig_mod", model)
