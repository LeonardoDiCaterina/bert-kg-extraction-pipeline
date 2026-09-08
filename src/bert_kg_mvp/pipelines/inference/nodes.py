import torch
import pandas as pd

def run_mvp_inference(processed_data: dict, tokenizer, trained_model, parameters: dict) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    trained_model.to(device)
    trained_model.eval()
    
    input_ids = processed_data["input_ids"].to(device)
    attention_mask = processed_data["attention_mask"].to(device)
    batch_size = input_ids.size(0)
    
    bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")
    decoder_input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
    
    max_gen_length = 40
    
    with torch.no_grad():
        for step in range(max_gen_length):
            logits = trained_model(input_ids, attention_mask, decoder_input_ids)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)
            
            # Early stopping if all sequences in batch generate EOS
            if (next_token_id == eos_token_id).all():
                break

    decoded_texts = tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=False)
    
    # Clean up output: take everything before the first EOS and append EOS
    clean_texts = [text.split("[EOS]")[0] + "[EOS]" for text in decoded_texts]
    
    return pd.DataFrame({
        "input_index": list(range(batch_size)),
        "generated_kg_text": clean_texts
    })
