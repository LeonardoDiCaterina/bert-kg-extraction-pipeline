import torch
from transformers import LogitsProcessor


class KGSchemaLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, valid_types, valid_relations):
        self.tokenizer = tokenizer
        self.pipe_id = tokenizer.convert_tokens_to_ids("|")
        self.eot_id = tokenizer.convert_tokens_to_ids("[EOT]")
        self.eos_id = tokenizer.convert_tokens_to_ids("[EOS]")

        self.valid_type_ids = {tokenizer.convert_tokens_to_ids(t) for t in valid_types}
        self.valid_relation_first_ids = {
            tokenizer.encode(r, add_special_tokens=False)[0] for r in valid_relations
        }

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            seq = input_ids[i]
            tokens_list = seq.tolist()

            if self.pipe_id in tokens_list:
                last_pipe_idx = max(
                    loc for loc, val in enumerate(tokens_list) if val == self.pipe_id
                )
                tokens_since_pipe = len(tokens_list) - last_pipe_idx - 1
            else:
                tokens_since_pipe = len(tokens_list)

            pipe_count = (seq == self.pipe_id).sum().item()
            mask = torch.full_like(scores[i], float("-inf"))

            # Simplified 3-tier slot check per triplet: Entity -> Type -> Relation
            if pipe_count == 0:
                # Expecting first pipe
                mask[self.pipe_id] = scores[i][self.pipe_id]
                scores[i] = mask
            elif pipe_count == 1:
                if tokens_since_pipe == 0:
                    # Expecting entity type
                    for tid in self.valid_type_ids:
                        mask[tid] = scores[i][tid]
                    scores[i] = mask
                else:
                    # Expecting second pipe after type
                    mask[self.pipe_id] = scores[i][self.pipe_id]
                    scores[i] = mask
            elif pipe_count == 2:
                # Expecting relation text, or [EOT]/[EOS]
                mask[self.eot_id] = scores[i][self.eot_id]
                mask[self.eos_id] = scores[i][self.eos_id]
                for rid in self.valid_relation_first_ids:
                    mask[rid] = scores[i][rid]
                scores[i] = mask
            else:
                # Reset pipe count behavior for multi-triple expansion if needed
                continue

        return scores
