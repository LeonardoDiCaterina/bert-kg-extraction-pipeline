import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class SetCriterion(nn.Module):
    """
    Computes the Bipartite Matching Loss between a set of predicted queries and ground truth triples.
    Inspired by DETR (DEtection TRansformer).
    """
    def __init__(self, num_relation_classes, num_entity_types, eos_coef=0.1):
        super().__init__()
        self.num_relation_classes = num_relation_classes
        self.num_entity_types = num_entity_types
        self.eos_coef = eos_coef # relative weight of the 'no_relation' (eos) class

        # Weights for the different parts of the loss
        self.weight_dict = {
            'loss_ce': 1.0,         # Relation classification
            'loss_type': 1.0,       # Entity type classification
            'loss_span': 1.0        # Pointer network span extraction
        }

        # Create an empty weight tensor for relation classes.
        # We down-weight the 'no_relation' class (index `num_relation_classes`) to handle class imbalance,
        # since most queries will map to 'no_relation'.
        empty_weight = torch.ones(self.num_relation_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    @torch.no_grad()
    def _get_src_permutation_idx(self, indices):
        # permutate predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @torch.no_grad()
    def _get_tgt_permutation_idx(self, indices):
        # permutate targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
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
        
        # Output shapes: [bs * num_queries, num_classes] / [bs * num_queries, seq_len]
        out_subj_start = outputs["subj_start_logits"].flatten(0, 1).softmax(-1)
        out_subj_end = outputs["subj_end_logits"].flatten(0, 1).softmax(-1)
        out_obj_start = outputs["obj_start_logits"].flatten(0, 1).softmax(-1)
        out_obj_end = outputs["obj_end_logits"].flatten(0, 1).softmax(-1)

        indices = []
        for b in range(bs):
            tgt_rels = targets[b]["relations"]
            if len(tgt_rels) == 0:
                indices.append((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)))
                continue

            # Relation class cost: -prob(target_class)
            cost_class = -out_prob[b * num_queries : (b + 1) * num_queries, tgt_rels]

            # Span cost: -prob(target_start) - prob(target_end)
            tgt_subj_spans = targets[b]["subj_spans"]
            tgt_obj_spans = targets[b]["obj_spans"]

            cost_span = -(
                out_subj_start[b * num_queries : (b + 1) * num_queries, tgt_subj_spans[:, 0]] +
                out_subj_end[b * num_queries : (b + 1) * num_queries, tgt_subj_spans[:, 1]] +
                out_obj_start[b * num_queries : (b + 1) * num_queries, tgt_obj_spans[:, 0]] +
                out_obj_end[b * num_queries : (b + 1) * num_queries, tgt_obj_spans[:, 1]]
            )

            # Total cost matrix for this batch element
            C = self.weight_dict['loss_ce'] * cost_class + self.weight_dict['loss_span'] * cost_span
            C = C.cpu().numpy()

            # Hungarian matching
            src_ind, tgt_ind = linear_sum_assignment(C)
            indices.append((torch.as_tensor(src_ind, dtype=torch.int64), torch.as_tensor(tgt_ind, dtype=torch.int64)))

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
        src_logits = outputs['rel_logits']
        
        # Target classes are 'no_relation' (idx = self.num_relation_classes) by default
        target_classes = torch.full(src_logits.shape[:2], self.num_relation_classes, dtype=torch.int64, device=src_logits.device)
        
        # For matched queries, use the actual ground truth relation class
        if len(tgt_idx[0]) > 0:
            target_classes_o = torch.cat([t["relations"][J] for t, (_, J) in zip(targets, indices)])
            target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.empty_weight)

        losses = {'loss_ce': loss_ce}

        if len(tgt_idx[0]) == 0:
            # If no ground truth triples exist, we only compute the loss_ce (to predict no_relation)
            return losses

        # 2. Entity Type Loss (only applied to matched queries)
        src_subj_type = outputs['subj_type_logits'][idx]
        src_obj_type = outputs['obj_type_logits'][idx]
        
        target_subj_type = torch.cat([t["subj_types"][J] for t, (_, J) in zip(targets, indices)])
        target_obj_type = torch.cat([t["obj_types"][J] for t, (_, J) in zip(targets, indices)])

        loss_type = (
            F.cross_entropy(src_subj_type, target_subj_type) + 
            F.cross_entropy(src_obj_type, target_obj_type)
        ) / 2
        losses['loss_type'] = loss_type

        # 3. Span (Pointer Network) Loss (only applied to matched queries)
        src_subj_start = outputs['subj_start_logits'][idx]
        src_subj_end = outputs['subj_end_logits'][idx]
        src_obj_start = outputs['obj_start_logits'][idx]
        src_obj_end = outputs['obj_end_logits'][idx]

        target_subj_spans = torch.cat([t["subj_spans"][J] for t, (_, J) in zip(targets, indices)])
        target_obj_spans = torch.cat([t["obj_spans"][J] for t, (_, J) in zip(targets, indices)])

        loss_span = (
            F.cross_entropy(src_subj_start, target_subj_spans[:, 0]) +
            F.cross_entropy(src_subj_end, target_subj_spans[:, 1]) +
            F.cross_entropy(src_obj_start, target_obj_spans[:, 0]) +
            F.cross_entropy(src_obj_end, target_obj_spans[:, 1])
        ) / 4
        losses['loss_span'] = loss_span

        return losses
