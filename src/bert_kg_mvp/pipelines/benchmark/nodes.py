import gc
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from bert_kg_mvp.models.architecture_2 import DynamicKGExtractor
from bert_kg_mvp.models.bipartite_loss import SetCriterion
from bert_kg_mvp.pipelines.data_prep.nodes import prepare_training_data
from bert_kg_mvp.utils import split_by_company


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
    model: DynamicKGExtractor,
    test_dataset: Dict[str, torch.Tensor],
    tokenizer: Any,
    no_relation_idx: int,
    device: torch.device,
    batch_size: int = 4,
) -> Tuple[float, float, float, float]:
    """
    Evaluates a trained DynamicKGExtractor model on test tensors.
    Returns:
        Tuple[float, float, float, float]: (precision, recall, f1, avg_latency_ms)
    """
    model.eval()
    input_ids = test_dataset["input_ids"]
    attention_mask = test_dataset["attention_mask"]
    gt_rels = test_dataset["relations"]
    gt_subj_spans = test_dataset["subj_spans"]
    gt_obj_spans = test_dataset["obj_spans"]

    total_samples = len(input_ids)
    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0

    true_positives, false_positives, false_negatives = 0, 0, 0
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
            subj_start_preds = torch.argmax(outputs["subj_start_logits"], dim=-1)
            subj_end_preds = torch.argmax(outputs["subj_end_logits"], dim=-1)
            obj_start_preds = torch.argmax(outputs["obj_start_logits"], dim=-1)
            obj_end_preds = torch.argmax(outputs["obj_end_logits"], dim=-1)

            for b in range(bs):
                pred_set = set()
                true_set = set()

                valid_gt = gt_rels[i + b] != no_relation_idx
                for r, ss, os in zip(
                    gt_rels[i + b][valid_gt],
                    gt_subj_spans[i + b][valid_gt],
                    gt_obj_spans[i + b][valid_gt],
                ):
                    subj_str = tokenizer.decode(
                        b_ids[b, ss[0] : ss[1] + 1], skip_special_tokens=True
                    ).strip()
                    obj_str = tokenizer.decode(
                        b_ids[b, os[0] : os[1] + 1], skip_special_tokens=True
                    ).strip()
                    true_set.add((subj_str, r.item(), obj_str))

                num_queries = outputs["rel_logits"].shape[1]
                for q in range(num_queries):
                    rel = rel_preds[b, q].item()
                    if rel != no_relation_idx:
                        s_start = min(
                            subj_start_preds[b, q].item(), subj_end_preds[b, q].item()
                        )
                        s_end = max(
                            subj_start_preds[b, q].item(), subj_end_preds[b, q].item()
                        )
                        o_start = min(
                            obj_start_preds[b, q].item(), obj_end_preds[b, q].item()
                        )
                        o_end = max(
                            obj_start_preds[b, q].item(), obj_end_preds[b, q].item()
                        )

                        pred_subj = tokenizer.decode(
                            b_ids[b, s_start : s_end + 1], skip_special_tokens=True
                        ).strip()
                        pred_obj = tokenizer.decode(
                            b_ids[b, o_start : o_end + 1], skip_special_tokens=True
                        ).strip()

                        if pred_subj and pred_obj:
                            pred_set.add((pred_subj, rel, pred_obj))

                true_positives += len(pred_set.intersection(true_set))
                false_positives += len(pred_set - true_set)
                false_negatives += len(true_set - pred_set)

            del b_ids, b_mask, outputs

    precision = true_positives / max((true_positives + false_positives), 1)
    recall = true_positives / max((true_positives + false_negatives), 1)
    f1 = 2 * (precision * recall) / max((precision + recall), 1e-9)
    avg_latency = float(sum(latencies) / max(len(latencies), 1))

    return precision, recall, f1, avg_latency


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
            if not param.requires_grad:
                continue
            if "encoder" in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
                
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": encoder_lr},
                {"params": decoder_params, "lr": learning_rate},
            ]
        )
        criterion = SetCriterion(
            num_relation_classes=num_relations, num_entity_types=num_ent_types
        ).to(device)

        train_ds = TensorDataset(
            train_tensors["input_ids"],
            train_tensors["attention_mask"],
            train_tensors["relations"],
            train_tensors["subj_types"],
            train_tensors["obj_types"],
            train_tensors["subj_spans"],
            train_tensors["obj_spans"],
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

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
        from collections import defaultdict
        import pandas as pd
        import os
        import copy
        
        history = []
        best_val_f1 = -1.0
        patience_counter = 0
        best_model_state = None
        val_interval_epochs = benchmark_params.get("val_interval_epochs", 5)
        patience = benchmark_params.get("early_stopping_patience", 3)
        
        for epoch in range(epochs):
            model.train()
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
                    optimizer.zero_grad()
                    
                total_loss += loss.item()
                for k, v in loss_dict.items():
                    epoch_losses[k] += v.item()

            avg_total = total_loss / len(train_loader)
            avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
            
            # Print at every epoch
            avg_components = " | ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
            print(f"  Epoch {epoch + 1}/{epochs} - Avg Total Loss: {avg_total:.4f} | {avg_components}")
                
            history_record = {"epoch": epoch + 1, "total_loss": avg_total}
            history_record.update(avg_losses)
            
            if (epoch + 1) % val_interval_epochs == 0:
                val_p, val_r, val_f1, _ = evaluate_model_on_test(
                    model=model,
                    test_dataset=val_tensors,
                    tokenizer=tokenizer,
                    no_relation_idx=no_relation_idx,
                    device=device,
                    batch_size=batch_size,
                )
                print(f"  >>> [Validation @ Epoch {epoch + 1}] F1: {val_f1:.4f} | Prec: {val_p:.4f} | Rec: {val_r:.4f}")
                history_record["val_f1"] = val_f1
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    
                model.train()
                
            history.append(history_record)
            
            if patience_counter >= patience:
                print(f"\n  [Early Stopping] No improvement for {patience} validation intervals. Stopping at Epoch {epoch + 1}.")
                break
            
        if best_model_state is not None:
            print(f"  [Restoring best weights] Reverting to model with Validation F1: {best_val_f1:.4f}")
            model.load_state_dict(best_model_state)
            
        # Save tracking history to CSV
        os.makedirs("data/08_reporting", exist_ok=True)
        safe_model_name = model_name.replace("/", "_")
        history_df = pd.DataFrame(history)
        history_df.to_csv(f"data/08_reporting/loss_history_{safe_model_name}.csv", index=False)

        total_train_sec = time.perf_counter() - start_train_time
        sec_per_epoch = total_train_sec / max(1, epochs)

        # 4. Evaluate on Test Split
        precision, recall, f1, avg_latency = evaluate_model_on_test(
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
                "test_f1": round(f1, 4),
                "test_precision": round(precision, 4),
                "test_recall": round(recall, 4),
                "latency_ms_per_doc": round(avg_latency, 2),
                "train_sec_per_epoch": round(sec_per_epoch, 2),
                "total_params_m": round(total_params / 1e6, 2),
                "trainable_params_m": round(trainable_params / 1e6, 2),
            }
        )

        print(
            f"Result for {display_name}: F1={f1:.4f} | Prec={precision:.4f} | Rec={recall:.4f} | "
            f"Latency={avg_latency:.2f}ms/doc | Train={sec_per_epoch:.2f}s/epoch"
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
