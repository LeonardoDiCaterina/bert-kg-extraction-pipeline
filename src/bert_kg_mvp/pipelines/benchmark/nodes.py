import copy
import gc
import os
import time
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from bert_kg_mvp.models.architecture_2 import DynamicKGExtractor
from bert_kg_mvp.models.bipartite_loss import SetCriterion
from bert_kg_mvp.pipelines.data_prep.nodes import prepare_training_data
from bert_kg_mvp.utils.splitting import split_by_company
from bert_kg_mvp.models import build_decoder


def split_dataset_node(
    teacher_triplets: pd.DataFrame,
    split_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Partitions the teacher-extracted triplets into train, validation, and test sets.
    Supports company-stratified splitting (unseen companies in val/test) or random splitting.
    """
    params = split_params or {}
    strategy = params.get("strategy", "company_stratified")
    train_ratio = params.get("train_ratio", 0.70)
    val_ratio = params.get("val_ratio", 0.15)
    test_ratio = params.get("test_ratio", 0.15)
    seed = params.get("random_seed", 42)

    if strategy == "company_stratified":
        train_df, val_df, test_df = split_by_company(
            teacher_triplets,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            company_col="doc_id"
            if "doc_id" in teacher_triplets.columns
            else teacher_triplets.columns[0],
        )
    else:
        # Standard randomized split
        shuffled = teacher_triplets.sample(frac=1, random_state=seed).reset_index(
            drop=True
        )
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_df = shuffled.iloc[:n_train].copy()
        val_df = shuffled.iloc[n_train : n_train + n_val].copy()
        test_df = shuffled.iloc[n_train + n_val :].copy()

    print(
        f"Dataset Split Completed ({strategy}): "
        f"Train={len(train_df)} samples, Val={len(val_df)} samples, Test={len(test_df)} samples."
    )

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def evaluate_model_on_test(
    model: torch.nn.Module,
    test_dataset: TensorDataset,
    tokenizer: Any,
    no_relation_idx: int,
    device: torch.device,
    batch_size: int = 16,
) -> Dict[str, float]:
    """
    Evaluates the model on the test dataset and returns Exact-Match Strict F1 and latency.
    """
    import numpy as np
    
    def jaccard_similarity(s1: str, s2: str) -> float:
        set1 = set(s1.lower().split())
        set2 = set(s2.lower().split())
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    model.eval()
    input_ids = test_dataset["input_ids"]
    attention_mask = test_dataset["attention_mask"]
    gt_rels = test_dataset["relations"]
    gt_subj_types = test_dataset["subj_types"]
    gt_obj_types = test_dataset["obj_types"]
    gt_subj_spans = test_dataset["subj_spans"]
    gt_obj_spans = test_dataset["obj_spans"]

    total_samples = len(input_ids)
    if total_samples == 0:
        return {}

    true_positives, false_positives, false_negatives = 0, 0, 0
    span_true_positives, span_false_positives, span_false_negatives = 0, 0, 0
    entity_tp, entity_fp, entity_fn = 0, 0, 0
    total_queries = 0
    no_rel_queries = 0
    
    type_correct_count = 0
    subj_type_correct_count = 0
    obj_type_correct_count = 0
    rel_correct_count = 0
    span_matched_count = 0
    
    per_rel_tp = Counter()
    per_rel_fp = Counter()
    per_rel_fn = defaultdict(int)

    total_error_distance = 0.0
    error_count = 0
    comp_errors = {"subj_span": 0, "subj_type": 0, "rel": 0, "obj_span": 0, "obj_type": 0}
    
    jaccard_scores = []
    latencies: List[float] = []

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            b_ids = input_ids[i : i + batch_size].to(device)
            b_mask = attention_mask[i : i + batch_size].to(device)
            bs = b_ids.size(0)

            t0 = time.perf_counter()
            outputs = model(b_ids, b_mask)
            if str(device) == "cuda":
                torch.cuda.synchronize()
            elif str(device) == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0 / max(1, bs))

            rel_preds = torch.argmax(outputs["rel_logits"], dim=-1)
            subj_type_preds = torch.argmax(outputs["subj_type_logits"], dim=-1)
            obj_type_preds = torch.argmax(outputs["obj_type_logits"], dim=-1)
            subj_slot_preds = torch.argmax(outputs["subj_slot_logits"], dim=-1)
            obj_slot_preds = torch.argmax(outputs["obj_slot_logits"], dim=-1)

            for b in range(bs):
                pred_set = set()
                true_set = set()

                valid_gt = gt_rels[i + b] != no_relation_idx
                for r, st, ot, ss, os in zip(
                    gt_rels[i + b][valid_gt],
                    gt_subj_types[i + b][valid_gt],
                    gt_obj_types[i + b][valid_gt],
                    gt_subj_spans[i + b][valid_gt],
                    gt_obj_spans[i + b][valid_gt],
                ):
                    subj_str = tokenizer.decode(
                        b_ids[b, ss[0] : ss[1] + 1], skip_special_tokens=True
                    ).strip()
                    obj_str = tokenizer.decode(
                        b_ids[b, os[0] : os[1] + 1], skip_special_tokens=True
                    ).strip()
                    true_set.add((subj_str, st.item(), r.item(), obj_str, ot.item()))

                num_queries = outputs["rel_logits"].shape[1]
                total_queries += num_queries
                for q in range(num_queries):
                    rel = rel_preds[b, q].item()
                    if rel == no_relation_idx:
                        no_rel_queries += 1
                    else:
                        # Subject span — decode only the specific tokens pointed to by active slots
                        s_slots = subj_slot_preds[b, q]
                        s_active = s_slots[s_slots < outputs["subj_slot_logits"].size(-1) - 1]
                        if len(s_active) == 0:
                            continue
                        s_token_indices = sorted(s_active.tolist())
                        s_token_ids = b_ids[b, s_token_indices]

                        # Object span — decode only the specific tokens pointed to by active slots
                        o_slots = obj_slot_preds[b, q]
                        o_active = o_slots[o_slots < outputs["obj_slot_logits"].size(-1) - 1]
                        if len(o_active) == 0:
                            continue
                        o_token_indices = sorted(o_active.tolist())
                        o_token_ids = b_ids[b, o_token_indices]

                        pred_subj = tokenizer.decode(
                            s_token_ids, skip_special_tokens=True
                        ).strip()
                        pred_obj = tokenizer.decode(
                            o_token_ids, skip_special_tokens=True
                        ).strip()
                        
                        stype = subj_type_preds[b, q].item()
                        otype = obj_type_preds[b, q].item()

                        if pred_subj and pred_obj:
                            pred_set.add((pred_subj, stype, rel, pred_obj, otype))

                tp_set = pred_set.intersection(true_set)
                fp_set = pred_set - true_set
                fn_set = true_set - pred_set
                
                true_positives += len(tp_set)
                false_positives += len(fp_set)
                false_negatives += len(fn_set)
                
                # Entity tracking (span text + entity type)
                pred_entities = set()
                for p in pred_set:
                    pred_entities.add((p[0], p[1]))  # Subject + SubjType
                    pred_entities.add((p[3], p[4]))  # Object + ObjType
                
                true_entities = set()
                for t in true_set:
                    true_entities.add((t[0], t[1]))
                    true_entities.add((t[3], t[4]))
                    
                entity_tp += len(pred_entities & true_entities)
                entity_fp += len(pred_entities - true_entities)
                entity_fn += len(true_entities - pred_entities)
                
                # Tier 1: Span matches
                pred_span_set = {(p[0], p[3]) for p in pred_set}
                true_span_set = {(t[0], t[3]) for t in true_set}
                span_true_positives += len(pred_span_set & true_span_set)
                span_false_positives += len(pred_span_set - true_span_set)
                span_false_negatives += len(true_span_set - pred_span_set)
                
                # Tier 2 & 3: Conditional on spans
                for pred in pred_set:
                    pred_spans = (pred[0], pred[3])
                    for gt in true_set:
                        gt_spans = (gt[0], gt[3])
                        if pred_spans == gt_spans:
                            span_matched_count += 1
                            if pred[1] == gt[1] and pred[4] == gt[4]:
                                type_correct_count += 1
                            if pred[1] == gt[1]:
                                subj_type_correct_count += 1
                            if pred[4] == gt[4]:
                                obj_type_correct_count += 1
                            if pred[2] == gt[2]:
                                rel_correct_count += 1
                            break
                            
                # Tier 4: Per Relation
                for t in tp_set:
                    per_rel_tp[t[2]] += 1
                for fp in fp_set:
                    per_rel_fp[fp[2]] += 1
                for fn in fn_set:
                    per_rel_fn[fn[2]] += 1
                
                for fp in fp_set:
                    if len(fn_set) == 0:
                        total_error_distance += 5.0
                        error_count += 1
                        comp_errors["subj_span"] += 1
                        comp_errors["subj_type"] += 1
                        comp_errors["rel"] += 1
                        comp_errors["obj_span"] += 1
                        comp_errors["obj_type"] += 1
                        continue
                        
                    min_dist = 6
                    best_fn = None
                    for fn in fn_set:
                        dist = sum(1 for c1, c2 in zip(fp, fn) if c1 != c2)
                        if dist < min_dist:
                            min_dist = dist
                            best_fn = fn
                            
                    total_error_distance += min_dist
                    error_count += 1
                    
                    if best_fn[0] != fp[0]: 
                        comp_errors["subj_span"] += 1
                        jaccard_scores.append(jaccard_similarity(fp[0], best_fn[0]))
                    if best_fn[1] != fp[1]: comp_errors["subj_type"] += 1
                    if best_fn[2] != fp[2]: comp_errors["rel"] += 1
                    if best_fn[3] != fp[3]: 
                        comp_errors["obj_span"] += 1
                        jaccard_scores.append(jaccard_similarity(fp[3], best_fn[3]))
                    if best_fn[4] != fp[4]: comp_errors["obj_type"] += 1

            del b_ids, b_mask, outputs

    precision = true_positives / max((true_positives + false_positives), 1)
    recall = true_positives / max((true_positives + false_negatives), 1)
    f1 = 2 * (precision * recall) / max((precision + recall), 1e-9)
    
    span_precision = span_true_positives / max((span_true_positives + span_false_positives), 1)
    span_recall = span_true_positives / max((span_true_positives + span_false_negatives), 1)
    span_f1 = 2 * (span_precision * span_recall) / max((span_precision + span_recall), 1e-9)
    
    type_accuracy = type_correct_count / max(span_matched_count, 1)
    subj_type_accuracy = subj_type_correct_count / max(span_matched_count, 1)
    obj_type_accuracy = obj_type_correct_count / max(span_matched_count, 1)
    rel_accuracy = rel_correct_count / max(span_matched_count, 1)
    
    entity_precision = entity_tp / max((entity_tp + entity_fp), 1)
    entity_recall = entity_tp / max((entity_tp + entity_fn), 1)
    no_rel_rate = no_rel_queries / max(total_queries, 1)
    
    avg_latency = float(sum(latencies) / max(len(latencies), 1))
    
    err_n = max(error_count, 1)
    avg_error_dist = total_error_distance / err_n
    comp_error_rates = {
        "subj_span_err_rate": comp_errors["subj_span"] / err_n,
        "subj_type_err_rate": comp_errors["subj_type"] / err_n,
        "rel_err_rate": comp_errors["rel"] / err_n,
        "obj_span_err_rate": comp_errors["obj_span"] / err_n,
        "obj_type_err_rate": comp_errors["obj_type"] / err_n,
    }
    
    # Calculate Mean, Median, Std Dev of Token Overlap Jaccard Similarity
    if jaccard_scores:
        j_mean = float(np.mean(jaccard_scores))
        j_median = float(np.median(jaccard_scores))
        j_std = float(np.std(jaccard_scores))
    else:
        j_mean = j_median = j_std = 0.0

    return {
        "strict_precision": precision,
        "strict_recall": recall,
        "strict_f1": f1,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "detected": true_positives + false_positives,
        "ground_truth": true_positives + false_negatives,
        "span_precision": span_precision,
        "span_recall": span_recall,
        "span_f1": span_f1,
        "tuple_precision": precision,
        "tuple_recall": recall,
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "type_accuracy": type_accuracy,
        "subj_type_accuracy": subj_type_accuracy,
        "obj_type_accuracy": obj_type_accuracy,
        "type_sample_size": span_matched_count,
        "rel_accuracy": rel_accuracy,
        "no_relation_rate": no_rel_rate,
        "per_rel_tp": dict(per_rel_tp),
        "per_rel_fp": dict(per_rel_fp),
        "per_rel_fn": dict(per_rel_fn),
        "mean_error_distance": avg_error_dist,
        "jaccard_mean": j_mean,
        "jaccard_median": j_median,
        "jaccard_std": j_std,
        "avg_latency_ms": avg_latency,
        **comp_error_rates,
    }


def run_decoder_benchmark(
    split_data: Dict[str, pd.DataFrame],
    benchmark_params: Dict[str, Any],
    data_prep_params: Dict[str, Any],
    schema_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Runs the multi-decoder benchmark using a fixed encoder.
    """
    decoder_configs = benchmark_params.get("decoder_configs", [])
    if not decoder_configs:
        # Fallback to single baseline config if missing
        decoder_configs = [{"decoder_type": "baseline", "display_name": "Baseline"}]
        
    encoder_model_name = benchmark_params.get("encoder", "nlpaueb/sec-bert-base")
    matmul_precision = benchmark_params.get("float32_matmul_precision", "high")
    if matmul_precision and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    schema = schema_params or {}
    relation_types = schema.get(
        "relation_types",
        ["has_metric", "produces", "operates_in", "reports_risk", "led_by"],
    )
    entity_types = schema.get(
        "entity_types",
        ["org", "person", "product", "segment", "fin_metric", "risk_factor", "event"],
    )
    num_relations = len(relation_types)
    num_ent_types = len(entity_types)
    no_relation_idx = num_relations

    models_config = benchmark_params.get(
        "models",
        [
            {"name": "bert-base-uncased", "display_name": "BERT Base"},
            {"name": "ProsusAI/finbert", "display_name": "FinBERT (ProsusAI)"},
            {"name": "nlpaueb/sec-bert-base", "display_name": "SEC-BERT (AUEB)"},
            {"name": "roberta-base", "display_name": "RoBERTa Base"},
        ],
    )

    epochs = benchmark_params.get("epochs", 5)
    loss_weights = benchmark_params.get(
        "loss_weights", 
        {"loss_ce": 1.0, "loss_type": 1.0, "loss_span": 1.0}
    )
    eos_coef = benchmark_params.get("eos_coef", 0.1)
    dynamic_unfreeze_epoch = benchmark_params.get("dynamic_unfreeze_epoch", 20)
    matcher_weights = benchmark_params.get(
        "matcher_weights", 
        {"loss_ce": 2.0, "loss_type": 1.0, "loss_span": 0.25}
    )
    learning_rate = benchmark_params.get("learning_rate", 5e-5)
    batch_size = benchmark_params.get("batch_size", 4)
    accum_steps = benchmark_params.get("gradient_accumulation_steps", 8)
    freeze_strategy = benchmark_params.get("freeze_strategy", "partial")
    unfrozen_top_layers = benchmark_params.get("unfrozen_top_layers", 4)
    num_queries = benchmark_params.get("num_queries", 15)
    decoder_num_layers = benchmark_params.get("decoder_num_layers", 4)
    d_model = benchmark_params.get("d_model", 768)

    train_df = split_data["train"]
    val_df = split_data["val"]
    test_df = split_data["test"]

    results: List[Dict[str, Any]] = []

    print("\n========================================================")
    print(f"Starting Decoder Benchmark ({len(decoder_configs)} variants configured) with encoder {encoder_model_name}")
    print(
        f"Device: {device} | Train samples: {len(train_df)} | Test samples: {len(test_df)}"
    )
    print("========================================================\n")

    # 1. Prepare data with encoder-specific tokenizer ONCE
    try:
        encoder_prep_params = dict(data_prep_params)
        train_tensors, tokenizer = prepare_training_data(
            train_df,
            data_prep_params=encoder_prep_params,
            training_params={"encoder_model_name": encoder_model_name},
            schema_params=schema_params,
        )
        val_tensors, _ = prepare_training_data(
            val_df,
            data_prep_params=encoder_prep_params,
            training_params={"encoder_model_name": encoder_model_name},
            schema_params=schema_params,
        )
        test_tensors, _ = prepare_training_data(
            test_df,
            data_prep_params=encoder_prep_params,
            training_params={"encoder_model_name": encoder_model_name},
            schema_params=schema_params,
        )
    except Exception as e:
        print(f"Error preparing data for {encoder_model_name}: {e}")
        return pd.DataFrame()

    for config_entry in decoder_configs:
        decoder_type = config_entry.get("decoder_type", "baseline")
        display_name = config_entry.get("display_name", decoder_type)
        
        cfg_num_queries = config_entry.get("num_queries", num_queries)
        cfg_queries_per_rel = config_entry.get("queries_per_rel", None)
        cfg_span_d_model = config_entry.get("span_d_model", 256)

        print(f"\n>>> Benchmarking Decoder: {display_name} ({decoder_type})")
        
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

        if decoder_type == "baseline":
            kwargs["num_queries"] = cfg_num_queries
        elif decoder_type == "typed":
            kwargs["queries_per_rel"] = cfg_queries_per_rel if cfg_queries_per_rel is not None else cfg_num_queries // num_relations
        elif decoder_type == "disentangled":
            kwargs["num_queries"] = cfg_num_queries
            kwargs["span_d_model"] = cfg_span_d_model

        model = build_decoder(decoder_type, **kwargs).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        encoder_lr = benchmark_params.get("encoder_learning_rate", 1e-5)
        encoder_params = []
        decoder_params = []
        for name, param in model.named_parameters():
            if "encoder" in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
                
        weight_decay = benchmark_params.get("weight_decay", 0.05)
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
            eos_coef=eos_coef,
            weight_dict=loss_weights,
            matcher_weight_dict=matcher_weights,
            queries_per_rel=kwargs.get("queries_per_rel") if decoder_type == "typed" else None
        ).to(device)

        from bert_kg_mvp.utils.dataset import AugmentedKGDataset
        prefix_end_token_id = tokenizer.convert_tokens_to_ids("]") if "]" in tokenizer.get_vocab() else None
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else pad_token_id
        
        train_ds = AugmentedKGDataset(
            tensors_dict=train_tensors,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            no_relation_idx=no_relation_idx,
            prefix_end_token_id=prefix_end_token_id,
            mask_prob=benchmark_params.get("mask_prob", 0.15),
            prefix_drop_prob=benchmark_params.get("prefix_drop_prob", 0.15),
            span_jitter_prob=benchmark_params.get("span_jitter_prob", 0.1)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        import math
        from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
        total_steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
        total_training_steps = total_steps_per_epoch * epochs
        num_warmup_steps = int(total_training_steps * 0.1) # 10% warmup
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
            num_cycles=3
        )
        
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema_decay = benchmark_params.get("ema_decay", 0.999)
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))

        # 3. Train
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

        # Optional torch.compile acceleration
        compile_model = benchmark_params.get("compile_model", False)
        if compile_model:
            if hasattr(torch, "compile"):
                compile_mode = benchmark_params.get("compile_mode", "default")
                try:
                    print(
                        f"Compiling {decoder_type} with torch.compile(mode='{compile_mode}')..."
                    )
                    model = torch.compile(model, mode=compile_mode)
                except Exception as exc:
                    print(
                        f"Warning: torch.compile failed for {decoder_type} ({exc}). Proceeding uncompiled."
                    )
            else:
                print(
                    "torch.compile is not available in this PyTorch version. Proceeding uncompiled."
                )

        start_train_time = time.perf_counter()
        
        history = []
        best_val_f1 = -1.0
        patience_counter = 0
        best_model_state = None
        val_interval_epochs = benchmark_params.get("val_interval_epochs", 5)
        patience = benchmark_params.get("early_stopping_patience", 3)
        
        for epoch in range(epochs):
            model.train()
            
            # Dynamic Unfreezing
            if epoch < dynamic_unfreeze_epoch:
                model.encoder.eval()
                for p in model.encoder.parameters():
                    p.requires_grad = False
            else:
                model.encoder.train()
                for p in model.encoder.parameters():
                    p.requires_grad = True
                    
            optimizer.zero_grad()
            total_loss = 0.0
            epoch_losses = defaultdict(float)
            
            for step, batch in enumerate(train_loader):
                batch = [b.to(device) for b in batch]
                b_ids, b_mask, b_rels, b_st, b_ot, b_ss, b_os = batch
                outputs = model(b_ids, b_mask)

                targets = []
                for i in range(b_ids.size(0)):
                    valid_idx = b_rels[i] != no_relation_idx
                    targets.append(
                        {
                            "relations": b_rels[i][valid_idx],
                            "subj_types": b_st[i][valid_idx],
                            "obj_types": b_ot[i][valid_idx],
                            "subj_spans": b_ss[i][valid_idx],
                            "obj_spans": b_os[i][valid_idx],
                        }
                    )

                loss_dict = criterion(outputs, targets)
                loss = sum(loss_dict.values())
                (loss / accum_steps).backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    scheduler.step()
                    ema_model.update_parameters(model)
                    optimizer.zero_grad()
                    
                total_loss += loss.item() if hasattr(loss, "item") else float(loss)
                for k, v in loss_dict.items():
                    epoch_losses[k] += v.item() if hasattr(v, "item") else float(v)

            avg_total = total_loss / len(train_loader)
            avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
            
            # Calculate L2 Norm of model parameters to show weight decay regularization effect
            l2_norm = sum(p.norm(2).item() for p in model.parameters() if p.requires_grad)
            
            # Print at every epoch
            avg_components = " | ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
            print(f"  Epoch {epoch + 1}/{epochs} - Avg Total Loss: {avg_total:.4f} | {avg_components} | l2_norm: {l2_norm:.2f}")
                
            history_record = {"epoch": epoch + 1, "total_loss": avg_total}
            history_record.update(avg_losses)
            
            if (epoch + 1) % val_interval_epochs == 0:
                val_metrics = evaluate_model_on_test(
                    model=ema_model,
                    test_dataset=val_tensors,
                    tokenizer=tokenizer,
                    no_relation_idx=no_relation_idx,
                    device=device,
                    batch_size=batch_size,
                )
                val_f1 = val_metrics.get("strict_f1", 0.0)
                print(
                    f"  >>> [Validation @ Epoch {epoch + 1}] "
                    f"Strict F1: {val_f1:.4f} | Span F1: {val_metrics.get('span_f1', 0.0):.4f} | "
                    f"TypeAcc: {val_metrics.get('type_accuracy', 0.0):.2f} | RelAcc: {val_metrics.get('rel_accuracy', 0.0):.2f} | "
                    f"TP: {val_metrics.get('tp', 0)} | Det: {val_metrics.get('detected', 0)} | "
                    f"GT: {val_metrics.get('ground_truth', 0)} | MED: {val_metrics.get('mean_error_distance', 0.0):.2f} | "
                    f"Jaccard (Mean/Med/Std): {val_metrics.get('jaccard_mean', 0.0):.2f} / {val_metrics.get('jaccard_median', 0.0):.2f} / {val_metrics.get('jaccard_std', 0.0):.2f}"
                )
                history_record["val_f1"] = val_f1
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    best_model_state = copy.deepcopy(ema_model.module.state_dict())
                else:
                    patience_counter += 1
                    
                model.train()
                
            history.append(history_record)
            
            if patience_counter >= patience:
                print(f"\n  [Early Stopping] No improvement for {patience} validation intervals. Stopping at Epoch {epoch + 1}.")
                break
            
        if best_model_state is not None:
            print(f"  [Restoring best weights] Reverting to model with Validation F1: {best_val_f1:.4f}")
            
            if compile_model and hasattr(torch, "compile"):
                # torch.compile adds '_orig_mod.' prefix to all parameters in the OptimizedModule.
                # Since we copied from ema_model, we need to add the prefix back.
                compiled_state = {}
                for k, v in best_model_state.items():
                    if not k.startswith("_orig_mod."):
                        compiled_state[f"_orig_mod.{k}"] = v
                    else:
                        compiled_state[k] = v
                model.load_state_dict(compiled_state)
            else:
                model.load_state_dict(best_model_state)
            
        # Save tracking history to CSV
        os.makedirs("data/08_reporting", exist_ok=True)
        safe_model_name = decoder_type.replace("/", "_")
        history_df = pd.DataFrame(history)
        history_df.to_csv(f"data/08_reporting/loss_history_{safe_model_name}.csv", index=False)

        total_train_sec = time.perf_counter() - start_train_time
        sec_per_epoch = total_train_sec / max(1, epochs)

        # 4. Evaluate on Test Split
        test_metrics = evaluate_model_on_test(
            model=model,
            test_dataset=test_tensors,
            tokenizer=tokenizer,
            no_relation_idx=no_relation_idx,
            device=device,
            batch_size=batch_size,
        )

        results.append(
            {
                "decoder_type": decoder_type,
                "display_name": display_name,
                "test_strict_f1": round(test_metrics.get("strict_f1", 0.0), 4),
                "test_span_f1": round(test_metrics.get("span_f1", 0.0), 4),
                "test_type_acc": round(test_metrics.get("type_accuracy", 0.0), 4),
                "test_rel_acc": round(test_metrics.get("rel_accuracy", 0.0), 4),
                "tp": test_metrics.get("tp", 0),
                "detected": test_metrics.get("detected", 0),
                "ground_truth": test_metrics.get("ground_truth", 0),
                "avg_error_dist": round(test_metrics.get("mean_error_distance", 0.0), 2),
                "latency_ms_per_doc": round(test_metrics.get("avg_latency_ms", 0.0), 2),
                "err_subj_span": round(test_metrics.get("subj_span_err_rate", 0.0), 2),
                "err_subj_type": round(test_metrics.get("subj_type_err_rate", 0.0), 2),
                "err_rel": round(test_metrics.get("rel_err_rate", 0.0), 2),
                "err_obj_span": round(test_metrics.get("obj_span_err_rate", 0.0), 2),
                "err_obj_type": round(test_metrics.get("obj_type_err_rate", 0.0), 2),
                "train_sec_per_epoch": round(sec_per_epoch, 2),
                "total_params_m": round(total_params / 1e6, 2),
                "trainable_params_m": round(trainable_params / 1e6, 2),
            }
        )

        print(
            f"Result for {display_name}: Strict F1={test_metrics.get('strict_f1', 0.0):.4f} | "
            f"Span F1={test_metrics.get('span_f1', 0.0):.4f} | TypeAcc={test_metrics.get('type_accuracy', 0.0):.4f} | "
            f"Latency={test_metrics.get('avg_latency_ms', 0.0):.2f}ms/doc | Train={sec_per_epoch:.2f}s/epoch"
        )

        # Cleanup memory before next encoder
        del (
            model,
            optimizer,
            criterion,
            train_ds,
            train_loader,
        )
        if str(device) == "cuda":
            torch.cuda.empty_cache()
        elif str(device) == "mps":
            torch.mps.empty_cache()
        gc.collect()

    results_df = pd.DataFrame(results)
    print("\n================ BENCHMARK SUMMARY TABLE ================")
    if not results_df.empty:
        print(results_df.to_string(index=False))
    print("=========================================================\n")

    return results_df


def run_encoder_benchmark(
    split_data: Dict[str, pd.DataFrame],
    benchmark_params: Dict[str, Any],
    data_prep_params: Dict[str, Any],
    schema_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Trains and benchmarks multiple encoder backbones on the company-stratified train/test splits.
    Reads list of models from `benchmark_params['models']`.
    """
    matmul_precision = benchmark_params.get("float32_matmul_precision", "high")
    if matmul_precision and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    schema = schema_params or {}
    relation_types = schema.get(
        "relation_types",
        ["has_metric", "produces", "operates_in", "reports_risk", "led_by"],
    )
    entity_types = schema.get(
        "entity_types",
        ["org", "person", "product", "segment", "fin_metric", "risk_factor", "event"],
    )
    num_relations = len(relation_types)
    num_ent_types = len(entity_types)
    no_relation_idx = num_relations

    models_config = benchmark_params.get(
        "models",
        [
            {"name": "bert-base-uncased", "display_name": "BERT Base"},
            {"name": "ProsusAI/finbert", "display_name": "FinBERT (ProsusAI)"},
            {"name": "nlpaueb/sec-bert-base", "display_name": "SEC-BERT (AUEB)"},
            {"name": "roberta-base", "display_name": "RoBERTa Base"},
        ],
    )

    epochs = benchmark_params.get("epochs", 5)
    loss_weights = benchmark_params.get(
        "loss_weights", 
        {"loss_ce": 1.0, "loss_type": 1.0, "loss_span": 1.0}
    )
    eos_coef = benchmark_params.get("eos_coef", 0.1)
    dynamic_unfreeze_epoch = benchmark_params.get("dynamic_unfreeze_epoch", 20)
    matcher_weights = benchmark_params.get(
        "matcher_weights", 
        {"loss_ce": 2.0, "loss_type": 1.0, "loss_span": 0.25}
    )
    learning_rate = benchmark_params.get("learning_rate", 5e-5)
    batch_size = benchmark_params.get("batch_size", 4)
    accum_steps = benchmark_params.get("gradient_accumulation_steps", 8)
    freeze_strategy = benchmark_params.get("freeze_strategy", "partial")
    unfrozen_top_layers = benchmark_params.get("unfrozen_top_layers", 4)
    num_queries = benchmark_params.get("num_queries", 15)
    decoder_num_layers = benchmark_params.get("decoder_num_layers", 4)
    d_model = benchmark_params.get("d_model", 768)

    train_df = split_data["train"]
    val_df = split_data["val"]
    test_df = split_data["test"]

    results: List[Dict[str, Any]] = []

    print("\n========================================================")
    print(f"Starting Multi-Encoder Benchmark ({len(models_config)} models configured)")
    print(
        f"Device: {device} | Train samples: {len(train_df)} | Test samples: {len(test_df)}"
    )
    print("========================================================\n")

    for model_entry in models_config:
        if isinstance(model_entry, str):
            model_name = model_entry
            display_name = model_entry
        else:
            model_name = model_entry.get("name")
            display_name = model_entry.get("display_name", model_name)

        print(f"\n>>> Benchmarking Encoder: {display_name} ({model_name})")

        # 1. Prepare data with encoder-specific tokenizer
        try:
            encoder_prep_params = dict(data_prep_params)
            train_tensors, tokenizer = prepare_training_data(
                train_df,
                data_prep_params=encoder_prep_params,
                training_params={"encoder_model_name": model_name},
                schema_params=schema,
            )
            val_tensors, _ = prepare_training_data(
                val_df,
                data_prep_params=encoder_prep_params,
                training_params={"encoder_model_name": model_name},
                schema_params=schema,
            )
            test_tensors, _ = prepare_training_data(
                test_df,
                data_prep_params=encoder_prep_params,
                training_params={"encoder_model_name": model_name},
                schema_params=schema,
            )
        except Exception as e:
            print(f"Error preparing data for {model_name}: {e}")
            continue

        # 2. Instantiate Model
        model = DynamicKGExtractor(
            encoder_model_name=model_name,
            d_model=d_model,
            num_layers=decoder_num_layers,
            num_queries=num_queries,
            num_relations=num_relations,
            num_ent_types=num_ent_types,
            freeze_strategy=freeze_strategy,
            unfrozen_top_layers=unfrozen_top_layers,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        encoder_lr = benchmark_params.get("encoder_learning_rate", 1e-5)
        encoder_params = []
        decoder_params = []
        for name, param in model.named_parameters():
            if "encoder" in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
                
        weight_decay = benchmark_params.get("weight_decay", 0.05)
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
            eos_coef=eos_coef,
            weight_dict=loss_weights,
            matcher_weight_dict=matcher_weights,
        ).to(device)

        from bert_kg_mvp.utils.dataset import AugmentedKGDataset
        prefix_end_token_id = tokenizer.convert_tokens_to_ids("]") if "]" in tokenizer.get_vocab() else None
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else pad_token_id
        
        train_ds = AugmentedKGDataset(
            tensors_dict=train_tensors,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            no_relation_idx=no_relation_idx,
            prefix_end_token_id=prefix_end_token_id,
            mask_prob=benchmark_params.get("mask_prob", 0.15),
            prefix_drop_prob=benchmark_params.get("prefix_drop_prob", 0.15),
            span_jitter_prob=benchmark_params.get("span_jitter_prob", 0.1)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        import math
        from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
        total_steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
        total_training_steps = total_steps_per_epoch * epochs
        num_warmup_steps = int(total_training_steps * 0.1) # 10% warmup
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
            num_cycles=3
        )
        
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema_decay = benchmark_params.get("ema_decay", 0.999)
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))

        # 3. Train
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

        # Optional torch.compile acceleration
        compile_model = benchmark_params.get("compile_model", False)
        if compile_model:
            if hasattr(torch, "compile"):
                compile_mode = benchmark_params.get("compile_mode", "default")
                try:
                    print(
                        f"Compiling {model_name} with torch.compile(mode='{compile_mode}')..."
                    )
                    model = torch.compile(model, mode=compile_mode)
                except Exception as exc:
                    print(
                        f"Warning: torch.compile failed for {model_name} ({exc}). Proceeding uncompiled."
                    )
            else:
                print(
                    "torch.compile is not available in this PyTorch version. Proceeding uncompiled."
                )

        start_train_time = time.perf_counter()
        
        history = []
        best_val_f1 = -1.0
        patience_counter = 0
        best_model_state = None
        val_interval_epochs = benchmark_params.get("val_interval_epochs", 5)
        patience = benchmark_params.get("early_stopping_patience", 3)
        
        from bert_kg_mvp.utils.checkpointer import ModelCheckpointer
        checkpoint_config = benchmark_params.get("checkpointing", {})
        use_checkpointing = checkpoint_config.get("enabled", False)
        
        if use_checkpointing:
            checkpointer = ModelCheckpointer(
                checkpoint_dir=checkpoint_config.get("dirpath", "data/06_models/checkpoints"),
                model_name=decoder_type,
                mode="max",
                save_top_k=checkpoint_config.get("save_top_k", 1),
                save_last=checkpoint_config.get("save_last", True),
                save_every_n_epochs=checkpoint_config.get("save_every_n_epochs", 0),
            )
        else:
            checkpointer = None
        
        
        for epoch in range(epochs):
            model.train()
            
            # Dynamic Unfreezing
            if epoch < dynamic_unfreeze_epoch:
                model.encoder.eval()
                for p in model.encoder.parameters():
                    p.requires_grad = False
            else:
                model.encoder.train()
                for p in model.encoder.parameters():
                    p.requires_grad = True
                    
            optimizer.zero_grad()
            total_loss = 0.0
            epoch_losses = defaultdict(float)
            
            for step, batch in enumerate(train_loader):
                batch = [b.to(device) for b in batch]
                b_ids, b_mask, b_rels, b_st, b_ot, b_ss, b_os = batch
                outputs = model(b_ids, b_mask)

                targets = []
                for i in range(b_ids.size(0)):
                    valid_idx = b_rels[i] != no_relation_idx
                    targets.append(
                        {
                            "relations": b_rels[i][valid_idx],
                            "subj_types": b_st[i][valid_idx],
                            "obj_types": b_ot[i][valid_idx],
                            "subj_spans": b_ss[i][valid_idx],
                            "obj_spans": b_os[i][valid_idx],
                        }
                    )

                loss_dict = criterion(outputs, targets)
                loss = sum(loss_dict.values())
                (loss / accum_steps).backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    scheduler.step()
                    ema_model.update_parameters(model)
                    optimizer.zero_grad()
                    
                total_loss += loss.item() if hasattr(loss, "item") else float(loss)
                for k, v in loss_dict.items():
                    epoch_losses[k] += v.item() if hasattr(v, "item") else float(v)

            avg_total = total_loss / len(train_loader)
            avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
            
            # Calculate L2 Norm of model parameters to show weight decay regularization effect
            l2_norm = sum(p.norm(2).item() for p in model.parameters() if p.requires_grad)
            
            # Print at every epoch
            avg_components = " | ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
            print(f"  Epoch {epoch + 1}/{epochs} - Avg Total Loss: {avg_total:.4f} | {avg_components} | l2_norm: {l2_norm:.2f}")
                
            history_record = {"epoch": epoch + 1, "total_loss": avg_total}
            history_record.update(avg_losses)
            
            if (epoch + 1) % val_interval_epochs == 0:
                val_metrics = evaluate_model_on_test(
                    model=ema_model,
                    test_dataset=val_tensors,
                    tokenizer=tokenizer,
                    no_relation_idx=no_relation_idx,
                    device=device,
                    batch_size=batch_size,
                )
                val_f1 = val_metrics.get("strict_f1", 0.0)
                print(
                    f"  >>> [Validation @ Epoch {epoch + 1}] "
                    f"Strict F1: {val_f1:.4f} | Span F1: {val_metrics.get('span_f1', 0.0):.4f} | "
                    f"TypeAcc: {val_metrics.get('type_accuracy', 0.0):.2f} | RelAcc: {val_metrics.get('rel_accuracy', 0.0):.2f} | "
                    f"TP: {val_metrics.get('tp', 0)} | Det: {val_metrics.get('detected', 0)} | "
                    f"GT: {val_metrics.get('ground_truth', 0)} | MED: {val_metrics.get('mean_error_distance', 0.0):.2f} | "
                    f"Jaccard (Mean/Med/Std): {val_metrics.get('jaccard_mean', 0.0):.2f} / {val_metrics.get('jaccard_median', 0.0):.2f} / {val_metrics.get('jaccard_std', 0.0):.2f}"
                )
                history_record["val_f1"] = val_f1
                
                if use_checkpointing:
                    is_best = checkpointer.save_checkpoint(
                        epoch=epoch + 1,
                        model=ema_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        metric_value=val_f1
                    )
                    # We still update best_val_f1 manually so the patience logic triggers
                    if is_best:
                        best_val_f1 = val_f1
                else:
                    is_best = val_f1 > best_val_f1
                    
                if is_best:
                    if not use_checkpointing:
                        best_val_f1 = val_f1
                    patience_counter = 0
                    best_model_state = copy.deepcopy(ema_model.module.state_dict())
                else:
                    patience_counter += 1
                    
                model.train()
                
            history.append(history_record)
            
            if patience_counter >= patience:
                print(f"\n  [Early Stopping] No improvement for {patience} validation intervals. Stopping at Epoch {epoch + 1}.")
                break
            
        if best_model_state is not None:
            print(f"  [Restoring best weights] Reverting to model with Validation F1: {best_val_f1:.4f}")
            
            if compile_model and hasattr(torch, "compile"):
                # torch.compile adds '_orig_mod.' prefix to all parameters in the OptimizedModule.
                # Since we copied from ema_model, we need to add the prefix back.
                compiled_state = {}
                for k, v in best_model_state.items():
                    if not k.startswith("_orig_mod."):
                        compiled_state[f"_orig_mod.{k}"] = v
                    else:
                        compiled_state[k] = v
                model.load_state_dict(compiled_state)
            else:
                model.load_state_dict(best_model_state)
            
        # Save tracking history to CSV
        os.makedirs("data/08_reporting", exist_ok=True)
        safe_model_name = model_name.replace("/", "_")
        history_df = pd.DataFrame(history)
        history_df.to_csv(f"data/08_reporting/loss_history_{safe_model_name}.csv", index=False)

        total_train_sec = time.perf_counter() - start_train_time
        sec_per_epoch = total_train_sec / max(1, epochs)

        # 4. Evaluate on Test Split
        test_metrics = evaluate_model_on_test(
            model=model,
            test_dataset=test_tensors,
            tokenizer=tokenizer,
            no_relation_idx=no_relation_idx,
            device=device,
            batch_size=batch_size,
        )

        results.append(
            {
                "model_name": model_name,
                "display_name": display_name,
                "test_strict_f1": round(test_metrics.get("strict_f1", 0.0), 4),
                "test_span_f1": round(test_metrics.get("span_f1", 0.0), 4),
                "test_type_acc": round(test_metrics.get("type_accuracy", 0.0), 4),
                "test_rel_acc": round(test_metrics.get("rel_accuracy", 0.0), 4),
                "tp": test_metrics.get("tp", 0),
                "detected": test_metrics.get("detected", 0),
                "ground_truth": test_metrics.get("ground_truth", 0),
                "avg_error_dist": round(test_metrics.get("mean_error_distance", 0.0), 2),
                "latency_ms_per_doc": round(test_metrics.get("avg_latency_ms", 0.0), 2),
                "err_subj_span": round(test_metrics.get("subj_span_err_rate", 0.0), 2),
                "err_subj_type": round(test_metrics.get("subj_type_err_rate", 0.0), 2),
                "err_rel": round(test_metrics.get("rel_err_rate", 0.0), 2),
                "err_obj_span": round(test_metrics.get("obj_span_err_rate", 0.0), 2),
                "err_obj_type": round(test_metrics.get("obj_type_err_rate", 0.0), 2),
                "train_sec_per_epoch": round(sec_per_epoch, 2),
                "total_params_m": round(total_params / 1e6, 2),
                "trainable_params_m": round(trainable_params / 1e6, 2),
            }
        )

        print(
            f"Result for {display_name}: Strict F1={test_metrics.get('strict_f1', 0.0):.4f} | "
            f"Span F1={test_metrics.get('span_f1', 0.0):.4f} | TypeAcc={test_metrics.get('type_accuracy', 0.0):.4f} | "
            f"Latency={test_metrics.get('avg_latency_ms', 0.0):.2f}ms/doc | Train={sec_per_epoch:.2f}s/epoch"
        )

        # Cleanup memory before next encoder
        del (
            model,
            optimizer,
            criterion,
            train_ds,
            train_loader,
            train_tensors,
            test_tensors,
        )
        if str(device) == "cuda":
            torch.cuda.empty_cache()
        elif str(device) == "mps":
            torch.mps.empty_cache()
        gc.collect()

    results_df = pd.DataFrame(results)
    print("\n================ BENCHMARK SUMMARY TABLE ================")
    if not results_df.empty:
        print(results_df.to_string(index=False))
    print("=========================================================\n")

    return results_df
