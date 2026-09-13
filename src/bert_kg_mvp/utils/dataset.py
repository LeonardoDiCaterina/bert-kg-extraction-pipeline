import torch
from torch.utils.data import Dataset
from typing import Dict

class AugmentedKGDataset(Dataset):
    """
    Dynamic dataset wrapper that applies pointer-safe data augmentations on the fly.
    """
    def __init__(
        self,
        tensors_dict: Dict[str, torch.Tensor],
        mask_token_id: int,
        pad_token_id: int,
        no_relation_idx: int,
        prefix_end_token_id: int = None,
        mask_prob: float = 0.15,
        prefix_drop_prob: float = 0.15,
        span_jitter_prob: float = 0.1,
    ):
        self.input_ids = tensors_dict["input_ids"]
        self.attention_mask = tensors_dict["attention_mask"]
        self.relations = tensors_dict["relations"]
        self.subj_types = tensors_dict["subj_types"]
        self.obj_types = tensors_dict["obj_types"]
        self.subj_spans = tensors_dict["subj_spans"]
        self.obj_spans = tensors_dict["obj_spans"]
        
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.no_relation_idx = no_relation_idx
        self.prefix_end_token_id = prefix_end_token_id
        self.mask_prob = mask_prob
        self.prefix_drop_prob = prefix_drop_prob
        self.span_jitter_prob = span_jitter_prob

    def __len__(self):
        return len(self.input_ids)
        
    def _jitter_span(self, span: torch.Tensor, max_idx: int) -> torch.Tensor:
        start, end = span[0].item(), span[1].item()
        
        # Randomly shift start by -1 or +1
        if self.span_jitter_prob > 0 and torch.rand(1).item() < self.span_jitter_prob:
            shift = 1 if torch.rand(1).item() > 0.5 else -1
            new_start = start + shift
            if 0 < new_start <= end:  # Don't cross 0 or the end index
                start = new_start
                
        # Randomly shift end by -1 or +1
        if self.span_jitter_prob > 0 and torch.rand(1).item() < self.span_jitter_prob:
            shift = 1 if torch.rand(1).item() > 0.5 else -1
            new_end = end + shift
            if start <= new_end < max_idx:  # Don't cross the start index or go out of bounds
                end = new_end
                
        return torch.tensor([start, end], dtype=torch.long)

    def __getitem__(self, idx: int):
        input_ids = self.input_ids[idx].clone()
        attention_mask = self.attention_mask[idx].clone()
        relations = self.relations[idx].clone()
        subj_types = self.subj_types[idx].clone()
        obj_types = self.obj_types[idx].clone()
        subj_spans = self.subj_spans[idx].clone()
        obj_spans = self.obj_spans[idx].clone()
        
        seq_length = (attention_mask == 1).sum().item()
        
        # 1. Prefix Dropping
        if self.prefix_drop_prob > 0 and torch.rand(1).item() < self.prefix_drop_prob:
            if self.prefix_end_token_id is not None:
                # Look for the prefix end token (']') in the first 30 tokens
                match_indices = (input_ids[:30] == self.prefix_end_token_id).nonzero(as_tuple=True)[0]
                if len(match_indices) > 0:
                    end_idx = match_indices[0].item()
                    # Mask out the prefix (keeping [CLS] at index 0 intact)
                    input_ids[1:end_idx+1] = self.mask_token_id

        # 2. Entity Masking & Span Jittering
        for i in range(len(relations)):
            if relations[i] != self.no_relation_idx:
                # Entity Masking - Subject
                if self.mask_prob > 0 and torch.rand(1).item() < self.mask_prob:
                    start, end = subj_spans[i]
                    input_ids[start:end+1] = self.mask_token_id
                
                # Entity Masking - Object
                if self.mask_prob > 0 and torch.rand(1).item() < self.mask_prob:
                    start, end = obj_spans[i]
                    input_ids[start:end+1] = self.mask_token_id
                    
                # Span Jittering - Subject
                subj_spans[i] = self._jitter_span(subj_spans[i], seq_length)
                
                # Span Jittering - Object
                obj_spans[i] = self._jitter_span(obj_spans[i], seq_length)
                
        return input_ids, attention_mask, relations, subj_types, obj_types, subj_spans, obj_spans
