from typing import Tuple
import torch


def align_entities_to_tokens(text: str, entity_str: str, tokenizer, input_ids: torch.Tensor) -> Tuple[int, int]:
    """
    Finds the start and end token indices of entity_str in the tokenized text using
    a sliding window search across the input_ids sequence.

    Returns:
        Tuple[int, int]: (start_token_idx, end_token_idx) inclusive, or (-1, -1) if not found.
    """
    ent_ids = tokenizer.encode(entity_str, add_special_tokens=False)
    if not ent_ids:
        return -1, -1

    seq = input_ids.tolist()
    len_ent = len(ent_ids)

    for i in range(len(seq) - len_ent + 1):
        if seq[i:i + len_ent] == ent_ids:
            return i, i + len_ent - 1

    return -1, -1
