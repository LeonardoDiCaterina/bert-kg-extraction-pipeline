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
    val_processed_dataset: Optional[Dict[str, torch.Tensor]] = None,
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
    id_to_rel = {i: r for i, r in enumerate(relation_types)}
    id_to_rel[no_relation_idx] = "no_relation"
    id_to_ent = {i: e for i, e in enumerate(entity_types)}

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
    
    eos_coef = training_params.get("eos_coef", 0.05)
    loss_weights = training_params.get("loss_weights", None)
    matcher_weights = training_params.get("matcher_weights", None)
    null_coef = training_params.get("null_coef", 0.3)
    num_token_slots = training_params.get("num_token_slots", 8)
    confidence_threshold = training_params.get("confidence_threshold", 0.5)

    criterion = SetCriterion(
        num_relation_classes=num_relations, 
        num_entity_types=num_ent_types,
        num_token_slots=num_token_slots,
        null_coef=null_coef,
        eos_coef=eos_coef,
        weight_dict=loss_weights,
        matcher_weight_dict=matcher_weights,
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

    epochs = training_params.get("epochs", 10)
    scheduler_type = training_params.get("scheduler", None)
    total_opt_steps = (len(dataloader) // accum_steps + (1 if len(dataloader) % accum_steps != 0 else 0)) * epochs
    warmup_epochs = training_params.get("warmup_epochs", 5)
    warmup_steps = (len(dataloader) // accum_steps + (1 if len(dataloader) % accum_steps != 0 else 0)) * warmup_epochs

    scheduler = None
    if scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_opt_steps,
        )
    elif scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_opt_steps,
        )

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

    from collections import defaultdict
    import pandas as pd
    import os
    
    history = []
    val_interval_epochs = training_params.get("val_interval_epochs", 5)
    
    from bert_kg_mvp.pipelines.benchmark.nodes import evaluate_model_on_test
    
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
            mode="max",  # Maximizing validation F1 score
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
                if scheduler is not None:
                    scheduler.step()
                ema_model.update_parameters(model)
                optimizer.zero_grad()

            total_loss += loss.item() if hasattr(loss, "item") else float(loss)
            for k, v in loss_dict.items():
                epoch_losses[k] += v.item() if hasattr(v, "item") else float(v)

            if step % 100 == 0:
                components_str = ", ".join([
                    f"{k}: {v.item():.4f}" if hasattr(v, "item") else f"{k}: {float(v):.4f}"
                    for k, v in loss_dict.items()
                ])
                total_val = loss.item() if hasattr(loss, "item") else float(loss)
                print(
                    f"Epoch {epoch + 1}/{epochs} | Batch {step}/{len(dataloader)} | Total: {total_val:.4f} | {components_str}"
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
        
        # Validation evaluation
        if val_processed_dataset is not None and (epoch + 1) % val_interval_epochs == 0:
            print(f"\n--- Running Validation (Epoch {epoch + 1}) ---")
            val_metrics = evaluate_model_on_test(
                model=ema_model.module,
                test_dataset=val_processed_dataset,
                tokenizer=tokenizer,
                no_relation_idx=no_relation_idx,
                device=device,
                batch_size=batch_size,
                confidence_threshold=confidence_threshold,
            )
            gt_count = val_metrics.get("ground_truth", 0)
            det_count = val_metrics.get("detected", 0)
            tp_count = val_metrics.get("tp", 0)
            fp_count = val_metrics.get("fp", 0)
            fn_count = val_metrics.get("fn", 0)
            print(f"Validation Triples (GT):     {gt_count}")
            print(f"Validation Triples (Extr):   {det_count} (TP: {tp_count}, FP: {fp_count}, FN: {fn_count})")
            print(f"Validation F1 (Exact Match): {val_metrics.get('strict_f1', 0.0):.4f}")
            print(f"Validation Tuple Prec:       {val_metrics.get('tuple_precision', 0.0):.4f}")
            print(f"Validation Tuple Rec:        {val_metrics.get('tuple_recall', 0.0):.4f}")
            print(f"Validation Entity Prec:      {val_metrics.get('entity_precision', 0.0):.4f}")
            print(f"Validation Entity Rec:       {val_metrics.get('entity_recall', 0.0):.4f}")
            print(f"Validation Span F1:          {val_metrics.get('span_f1', 0.0):.4f}")
            print(f"Validation Type Acc:         {val_metrics.get('type_accuracy', 0.0):.4f}")
            print(f"Validation No-Rel Rate:      {val_metrics.get('no_relation_rate', 0.0):.4f}")
            print(f"Validation Active Query Rate:{val_metrics.get('active_query_rate', 0.0):.4f}")
            print(f"Validation Mean FG Prob:     {val_metrics.get('mean_fg_prob', 0.0):.4f}")
            print(f"Validation Max FG Prob:      {val_metrics.get('max_fg_prob', 0.0):.4f}")
            print(f"Validation Mean No-Rel Prob: {val_metrics.get('mean_no_rel_prob', 0.0):.4f}")
            print(f"Validation MED:              {val_metrics.get('mean_error_distance', 0.0):.4f}")
            print(f"Validation Jaccard Mean:     {val_metrics.get('jaccard_mean', 0.0):.4f}")
            print(f"Validation Jaccard Median:   {val_metrics.get('jaccard_median', 0.0):.4f}")
            print(f"Validation Jaccard Std:      {val_metrics.get('jaccard_std', 0.0):.4f}")

            # Qualitative Triples Samples
            sample_triples = val_metrics.get("sample_triples", [])
            if sample_triples:
                print("  --- Sample Extracted vs Ground Truth Triples ---")
                for s in sample_triples[:3]:
                    print(f"  [Chunk {s['sample_idx']}]")
                    if s["true_triples"]:
                        for t in s["true_triples"]:
                            r_name = id_to_rel.get(t[2], str(t[2]))
                            st_name = id_to_ent.get(t[1], str(t[1]))
                            ot_name = id_to_ent.get(t[4], str(t[4]))
                            print(f"    ✅ GT:   ({t[0]!r} [{st_name}], {r_name}, {t[3]!r} [{ot_name}])")
                    else:
                        print("    ✅ GT:   <None>")
                    if s["pred_triples"]:
                        for p in s["pred_triples"]:
                            r_name = id_to_rel.get(p[2], str(p[2]))
                            st_name = id_to_ent.get(p[1], str(p[1]))
                            ot_name = id_to_ent.get(p[4], str(p[4]))
                            status = "MATCH" if p in s.get("tp_triples", []) else "FP"
                            print(f"    🤖 Extr [{status}]: ({p[0]!r} [{st_name}], {r_name}, {p[3]!r} [{ot_name}])")
                    else:
                        print("    🤖 Extr: <None>")
            print()
            
            history_record["val_gt_triples"] = gt_count
            history_record["val_extr_triples"] = det_count
            history_record["val_tp_triples"] = tp_count
            history_record["val_strict_f1"] = val_metrics.get('strict_f1', 0.0)
            history_record["val_tuple_prec"] = val_metrics.get('tuple_precision', 0.0)
            history_record["val_tuple_rec"] = val_metrics.get('tuple_recall', 0.0)
            history_record["val_entity_prec"] = val_metrics.get('entity_precision', 0.0)
            history_record["val_entity_rec"] = val_metrics.get('entity_recall', 0.0)
            history_record["val_no_rel_rate"] = val_metrics.get('no_relation_rate', 0.0)
            history_record["val_active_query_rate"] = val_metrics.get('active_query_rate', 0.0)
            history_record["val_mean_fg_prob"] = val_metrics.get('mean_fg_prob', 0.0)
            history_record["val_max_fg_prob"] = val_metrics.get('max_fg_prob', 0.0)
            history_record["val_mean_no_rel_prob"] = val_metrics.get('mean_no_rel_prob', 0.0)
            history_record["val_span_f1"] = val_metrics.get('span_f1', 0.0)
            history_record["val_type_accuracy"] = val_metrics.get('type_accuracy', 0.0)
            history_record["val_med"] = val_metrics.get('mean_error_distance', 0.0)
            history_record["val_jaccard_mean"] = val_metrics.get('jaccard_mean', 0.0)
            history_record["val_jaccard_median"] = val_metrics.get('jaccard_median', 0.0)
            history_record["val_jaccard_std"] = val_metrics.get('jaccard_std', 0.0)
            
            if use_checkpointing:
                checkpointer.save_checkpoint(
                    epoch=epoch + 1,
                    model=ema_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metric_value=val_metrics.get('strict_f1', 0.0)
                )
        elif use_checkpointing:
            checkpointer.save_checkpoint(
                epoch=epoch + 1,
                model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                metric_value=avg_total
            )

        history.append(history_record)

    # Save tracking history to CSV
    os.makedirs("data/08_reporting", exist_ok=True)
    safe_model_name = encoder_model_name.replace("/", "_")
    history_df = pd.DataFrame(history)
    history_df.to_csv(f"data/08_reporting/loss_history_{safe_model_name}.csv", index=False)
    print(f"Saved loss history to data/08_reporting/loss_history_{safe_model_name}.csv")

    # Load EMA weights into the raw model before returning
    model.load_state_dict(ema_model.module.state_dict())
    
    # Return underlying uncompiled model's state dict (for clean serialization)
    return getattr(model, "_orig_mod", model).state_dict()
