import gc
from typing import Any, Dict, Optional
import torch
from torch.utils.data import TensorDataset, DataLoader
from bert_kg_mvp.models.bipartite_loss import SetCriterion
from bert_kg_mvp.models.architecture_2 import DynamicKGExtractor


def train_model(
    processed_dataset: Dict[str, torch.Tensor],
    tokenizer: Any,
    training_params: Dict[str, Any],
    schema_params: Optional[Dict[str, Any]] = None,
) -> DynamicKGExtractor:
    """
    Trains the DynamicKGExtractor model using DETR-style bipartite matching loss.
    Supports either namespaced params (`params:training` + `params:schema`) or a legacy single dict.
    """
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

    model = DynamicKGExtractor(
        encoder_model_name=encoder_model_name,
        d_model=d_model,
        num_layers=decoder_num_layers,
        num_queries=num_queries,
        num_relations=num_relations,
        num_ent_types=num_ent_types,
        freeze_strategy=freeze_strategy,
        unfrozen_top_layers=unfrozen_top_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_params.get("learning_rate", 5e-5)
    )
    criterion = SetCriterion(
        num_relation_classes=num_relations, num_entity_types=num_ent_types
    ).to(device)

    dataset = TensorDataset(
        processed_dataset["input_ids"],
        processed_dataset["attention_mask"],
        processed_dataset["relations"],
        processed_dataset["subj_types"],
        processed_dataset["obj_types"],
        processed_dataset["subj_spans"],
        processed_dataset["obj_spans"],
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

    for epoch in range(epochs):
        total_loss = 0.0
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
                optimizer.zero_grad()

            total_loss += loss.item()

            if step % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | Batch {step}/{len(dataloader)} | Loss: {loss.item():.4f}"
                )

            if str(device) == "mps":
                torch.mps.empty_cache()
            del outputs, loss, loss_dict, batch
            if step % 50 == 0:
                gc.collect()

        print(
            f"Epoch {epoch + 1}/{epochs} - Avg Loss: {(total_loss / len(dataloader)):.4f}"
        )

    # Return underlying uncompiled model if wrapped (for clean serialization)
    return getattr(model, "_orig_mod", model)
