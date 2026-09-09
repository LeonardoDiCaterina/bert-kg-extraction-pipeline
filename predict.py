import sys
import torch
from pathlib import Path
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession

# 1. Import your working visualizer
from visualize_graph import parse_and_visualize

def predict_triplets(text: str):
    bootstrap_project(Path.cwd())
    with KedroSession.create() as session:
        context = session.load_context()
        tokenizer = context.catalog.load("kg_tokenizer")
        model = context.catalog.load("trained_model")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device)
    model.eval()

    encodings = tokenizer([text], padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")
    decoder_input_ids = torch.full((1, 1), bos_token_id, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(60):
            logits = model(input_ids, attention_mask, decoder_input_ids)
            next_logits = logits[:, -1, :].clone()

            # Slight repetition penalty
            for token_id in set(decoder_input_ids[0].tolist()):
                if token_id not in [tokenizer.convert_tokens_to_ids(t) for t in ['<triplet>', '<subj_type>', '<relation>', '<obj>', '<obj_type>', 'entity']]:
                    next_logits[0, token_id] /= 1.2

            next_token_id = torch.argmax(next_logits, dim=-1).unsqueeze(-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)

            if next_token_id.item() == eos_token_id:
                break

    output = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=False)
    print(f"\n[INPUT]:  {text}")
    print(f"[OUTPUT]: {output.strip()}\n")
    
    # 2. Automatically plot the graph
    print("Drawing Knowledge Graph...")
    parse_and_visualize(output)

if __name__ == "__main__":
    test_text = sys.argv[1] if len(sys.argv) > 1 else "Elon Musk founded SpaceX. SpaceX is located in California."
    predict_triplets(test_text)
