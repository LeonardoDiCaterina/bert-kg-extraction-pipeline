from typing import Tuple
import torch


def align_entities_to_tokens(
    text: str, entity_str: str, tokenizer, input_ids: torch.Tensor, encodings=None
) -> Tuple[int, int]:
    """
    Finds the start and end token indices of entity_str in the tokenized text.
    Uses character-level mapping for BPE/case-sensitive tokenizers if a Fast tokenizer
    is available, otherwise falls back to a sliding window search.

    Returns:
        Tuple[int, int]: (start_token_idx, end_token_idx) inclusive, or (-1, -1) if not found.
    """
    if not entity_str:
        return -1, -1

    # 1. Fast Tokenizer Character Mapping (Robust against BPE spacing and Case-Sensitivity)
    if encodings is not None and getattr(tokenizer, "is_fast", False):
        char_start = text.lower().find(entity_str.lower())
        if char_start != -1:
            char_end = char_start + len(entity_str) - 1
            
            token_start = encodings.char_to_token(0, char_start)
            token_end = encodings.char_to_token(0, char_end)
            
            # If boundaries fall on spaces/punctuation, expand search inward/outward
            if token_end is None:
                for i in range(char_end, char_start - 1, -1):
                    t = encodings.char_to_token(0, i)
                    if t is not None:
                        token_end = t
                        break
            if token_start is None:
                for i in range(char_start, char_end + 1):
                    t = encodings.char_to_token(0, i)
                    if t is not None:
                        token_start = t
                        break
                        
            if token_start is not None and token_end is not None:
                return token_start, token_end

    # 2. Fallback: Token-based sliding window (Best for WordPiece / uncased)
    seq = input_ids.tolist()
    candidates = [
        tokenizer.encode(entity_str, add_special_tokens=False),
        tokenizer.encode(f" {entity_str}", add_special_tokens=False),
        tokenizer.encode(entity_str.lower(), add_special_tokens=False),
    ]

    for ent_ids in candidates:
        if not ent_ids:
            continue
        len_ent = len(ent_ids)
        for i in range(len(seq) - len_ent + 1):
            if seq[i : i + len_ent] == ent_ids:
                return i, i + len_ent - 1

    return -1, -1
