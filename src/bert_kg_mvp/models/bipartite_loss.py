import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

def multiclass_focal_loss(inputs, targets, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
    """
    Focal loss for multi-class classification.
    inputs: [..., C] (e.g. [batch, queries, classes])
    targets: [...] (e.g. [batch, queries])
    """
    C = inputs.size(-1)
    inputs_flat = inputs.reshape(-1, C)
    targets_flat = targets.reshape(-1)

    # Standard Cross Entropy computes -alpha[y] * log(p_t)
    ce_loss = F.cross_entropy(
        inputs_flat, 
        targets_flat, 
        weight=alpha, 
        reduction='none', 
        label_smoothing=label_smoothing
    )

    # Get the probabilities of the target classes (p_t)
    probs = F.softmax(inputs_flat, dim=-1)
    pt = probs.gather(-1, targets_flat.unsqueeze(-1)).squeeze(-1)

    # Compute focal term
    focal_term = (1 - pt) ** gamma
    loss = focal_term * ce_loss

    if reduction == 'mean':
        if alpha is not None:
            # Emulate PyTorch's cross_entropy weighted mean behavior: divide by sum of weights
            alpha_t = alpha.gather(0, targets_flat)
            return loss.sum() / alpha_t.sum().clamp(min=1e-5)
        else:
            return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    
    return loss.view(targets.shape)


class SetCriterion(nn.Module):
    """
    Computes the Bipartite Matching Loss between a set of predicted queries and ground truth triples.
    Inspired by DETR (DEtection TRansformer).
    """

    def __init__(
        self,
        num_relation_classes,
        num_entity_types,
        num_token_slots=8,
        null_coef=0.3,
        eos_coef=0.1,
        weight_dict=None,
        matcher_weight_dict=None,
        queries_per_rel=None,
    ):
        super().__init__()
        self.num_relation_classes = num_relation_classes
        self.num_entity_types = num_entity_types
        self.num_token_slots = num_token_slots
        self.null_coef = null_coef
        self.eos_coef = eos_coef  # relative weight of the 'no_relation' (eos) class
        self.queries_per_rel = queries_per_rel

        # Weights for the different parts of the loss
        if weight_dict is None:
            self.weight_dict = {
                "loss_ce": 1.0,  # Relation classification
                "loss_type": 1.0,  # Entity type classification
                "loss_token": 1.5,  # Token slot assignment
                "loss_contig": 0.3,  # Contiguity regularizer
            }
        else:
            self.weight_dict = weight_dict
            
        if matcher_weight_dict is None:
            self.matcher_weight_dict = {
                "loss_ce": 1.0,
                "loss_type": 1.0,
                "loss_span": 1.0,
            }
        else:
            self.matcher_weight_dict = matcher_weight_dict

        # Create an empty weight tensor for relation classes.
        # We down-weight the 'no_relation' class (index `num_relation_classes`) to handle class imbalance,
        # since most queries will map to 'no_relation'.
        empty_weight = torch.ones(self.num_relation_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    @torch.no_grad()
    def _get_src_permutation_idx(self, indices):
        # permutate predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @torch.no_grad()
    def _get_tgt_permutation_idx(self, indices):
        # permutate targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @torch.no_grad()
    def match(self, outputs, targets):
        """
        Performs the bipartite matching.
        """
        bs, num_queries = outputs["rel_logits"].shape[:2]

        # We flatten to compute the cost matrix across all batch elements
        out_prob = outputs["rel_logits"].flatten(0, 1).softmax(-1)
        out_subj_type = outputs["subj_type_logits"].flatten(0, 1).softmax(-1)
        out_obj_type = outputs["obj_type_logits"].flatten(0, 1).softmax(-1)

        # Output shapes: [bs * num_queries, K, seq_len+1]
        out_subj_slot = outputs["subj_slot_logits"].flatten(0, 1).softmax(-1)
        out_obj_slot = outputs["obj_slot_logits"].flatten(0, 1).softmax(-1)

        indices = []
        for b in range(bs):
            tgt_rels = targets[b]["relations"]
            if len(tgt_rels) == 0:
                indices.append(
                    (
                        torch.tensor([], dtype=torch.int64),
                        torch.tensor([], dtype=torch.int64),
                    )
                )
                continue

            # Relation class cost: -prob(target_class)
            cost_class = -out_prob[b * num_queries : (b + 1) * num_queries, tgt_rels]

            # Type cost: -prob(target_subj_type) - prob(target_obj_type)
            tgt_subj_types = targets[b]["subj_types"]
            tgt_obj_types = targets[b]["obj_types"]
            cost_type = -(
                out_subj_type[b * num_queries : (b + 1) * num_queries, tgt_subj_types]
                + out_obj_type[b * num_queries : (b + 1) * num_queries, tgt_obj_types]
            )

            # Span cost proxy: max prob ANY slot assigns to each GT token
            tgt_subj_spans = targets[b]["subj_spans"]
            tgt_obj_spans = targets[b]["obj_spans"]

            q_subj = out_subj_slot[b * num_queries : (b + 1) * num_queries]  # [Q, K, L+1]
            q_obj = out_obj_slot[b * num_queries : (b + 1) * num_queries]

            cost_span = torch.zeros(num_queries, len(tgt_rels), device=out_subj_slot.device)
            for g in range(len(tgt_rels)):
                gt_subj_tokens = torch.arange(tgt_subj_spans[g, 0], tgt_subj_spans[g, 1] + 1, device=out_subj_slot.device)
                gt_obj_tokens = torch.arange(tgt_obj_spans[g, 0], tgt_obj_spans[g, 1] + 1, device=out_subj_slot.device)
                
                # Coverage proxy
                subj_cov = q_subj[:, :, gt_subj_tokens].max(dim=1).values.sum(dim=-1)
                obj_cov = q_obj[:, :, gt_obj_tokens].max(dim=1).values.sum(dim=-1)
                cost_span[:, g] = -(subj_cov + obj_cov)

            # Total cost matrix for this batch element
            C = (
                self.matcher_weight_dict["loss_ce"] * cost_class
                + self.matcher_weight_dict["loss_type"] * cost_type
                + self.matcher_weight_dict["loss_span"] * cost_span
            )
            C = C.cpu().numpy()

            # For Option A: Relation-Typed Queries
            if getattr(self, "queries_per_rel", None) is not None:
                num_q = C.shape[0]
                for t_idx, rel_id in enumerate(tgt_rels):
                    r = rel_id.item()
                    if r < self.num_relation_classes:
                        q_start = r * self.queries_per_rel
                        q_end = (r + 1) * self.queries_per_rel
                        # Add a huge penalty to queries not designated for this relation type
                        # We don't use np.inf to avoid scipy crashes if GT count > queries_per_rel
                        mask = np.ones(num_q, dtype=bool)
                        mask[q_start:q_end] = False
                        C[mask, t_idx] += 1e6

            # Hungarian matching
            src_ind, tgt_ind = linear_sum_assignment(C)
            indices.append(
                (
                    torch.as_tensor(src_ind, dtype=torch.int64),
                    torch.as_tensor(tgt_ind, dtype=torch.int64),
                )
            )

        return indices

    def forward(self, outputs, targets):
        """
        Calculates the Bipartite Matching Loss.
        """
        # Retrieve the matching between the outputs and the targets
        indices = self.match(outputs, targets)

        # Extract matching indices
        idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # 1. Relation Classification Loss (applied to all queries, unmatched are forced to 'no_relation')
        src_logits = outputs["rel_logits"]

        # Target classes are 'no_relation' (idx = self.num_relation_classes) by default
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_relation_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )

        # For matched queries, use the actual ground truth relation class
        if len(tgt_idx[0]) > 0:
            target_classes_o = torch.cat(
                [t["relations"][J] for t, (_, J) in zip(targets, indices)]
            )
            target_classes[idx] = target_classes_o

        loss_ce = multiclass_focal_loss(
            src_logits, target_classes, alpha=self.empty_weight, gamma=2.0, label_smoothing=0.1
        )

        losses = {"loss_ce": loss_ce * self.weight_dict["loss_ce"]}

        if len(tgt_idx[0]) == 0:
            # If no ground truth triples exist, we only compute the loss_ce (to predict no_relation)
            return losses

        # 2. Entity Type Loss (only applied to matched queries)
        src_subj_type = outputs["subj_type_logits"][idx]
        src_obj_type = outputs["obj_type_logits"][idx]

        target_subj_type = torch.cat(
            [t["subj_types"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_obj_type = torch.cat(
            [t["obj_types"][J] for t, (_, J) in zip(targets, indices)]
        )

        loss_type = (
            F.cross_entropy(src_subj_type, target_subj_type, label_smoothing=0.1)
            + F.cross_entropy(src_obj_type, target_obj_type, label_smoothing=0.1)
        ) / 2
        losses["loss_type"] = loss_type * self.weight_dict["loss_type"]

        # 3. Token Slot Assignment Loss (only on matched queries)
        loss_token, loss_contig = self._compute_token_slot_loss(
            outputs, targets, indices, idx, tgt_idx
        )
        losses["loss_token"] = loss_token * self.weight_dict["loss_token"]
        losses["loss_contig"] = loss_contig * self.weight_dict["loss_contig"]

        return losses

    def _compute_token_slot_loss(self, outputs, targets, indices, idx, tgt_idx):
        K = self.num_token_slots
        null_idx_val = outputs["subj_slot_logits"].size(-1) - 1

        total_token_loss = 0.0
        total_contig_loss = 0.0
        num_matched = 0

        for b, (src_indices, tgt_indices) in enumerate(indices):
            if len(src_indices) == 0:
                continue

            for i in range(len(src_indices)):
                q = src_indices[i]
                t = tgt_indices[i]

                for entity_key, span_key in [
                    ("subj_slot_logits", "subj_spans"),
                    ("obj_slot_logits", "obj_spans"),
                ]:
                    slot_logits = outputs[entity_key][b, q]   # [K, L+1]
                    span = targets[b][span_key][t]             # [start, end]
                    gt_tokens = torch.arange(span[0], span[1] + 1, device=slot_logits.device)

                    # Truncate to K if entity is longer than K tokens
                    gt_tokens = gt_tokens[:K]
                    T = len(gt_tokens)

                    # Build K targets: T real tokens + (K-T) nulls
                    inner_targets = torch.full((K,), null_idx_val, device=slot_logits.device, dtype=torch.long)
                    inner_targets[:T] = gt_tokens

                    # Inner cost matrix [K, K]
                    slot_probs = F.softmax(slot_logits, dim=-1)
                    inner_cost = torch.zeros(K, K, device=slot_logits.device)
                    for s in range(K):
                        for t_inner in range(K):
                            inner_cost[s, t_inner] = -torch.log(
                                slot_probs[s, inner_targets[t_inner]] + 1e-8
                            )

                    # Solve inner Hungarian
                    row_ind, col_ind = linear_sum_assignment(inner_cost.detach().cpu().numpy())

                    # Compute CE loss on matched pairs
                    for s, t_inner in zip(row_ind, col_ind):
                        target_token = inner_targets[t_inner]
                        weight = self.null_coef if t_inner >= T else 1.0
                        total_token_loss += weight * F.cross_entropy(
                            slot_logits[s].unsqueeze(0), target_token.unsqueeze(0)
                        )

                    num_matched += K

                    # Contiguity penalty on active slots
                    positions = torch.arange(null_idx_val, device=slot_logits.device, dtype=torch.float)
                    active_probs = slot_probs[:, :null_idx_val]   # [K, L]
                    null_prob = slot_probs[:, null_idx_val]         # [K]
                    expected_pos = (active_probs * positions).sum(-1)  # [K]
                    active_mask = (null_prob < 0.5).float()

                    masked_pos = expected_pos + (1 - active_mask) * 1e6
                    sorted_pos, _ = torch.sort(masked_pos)
                    n_active = int(active_mask.sum().item())
                    if n_active >= 2:
                        gaps = sorted_pos[1:n_active] - sorted_pos[:n_active-1]
                        total_contig_loss += ((gaps - 1.0) ** 2).mean()

        loss_token = total_token_loss / max(num_matched, 1)
        loss_contig = total_contig_loss / max(num_matched // K, 1)

        return loss_token, loss_contig
