from __future__ import annotations

import argparse
import json
import torch

from data.dataset import SequenceTokenizer
from models.conditional_generator import EnvConditionedGenerator
from utils.io import load_config


def sample_sequence(model, tokenizer, env, max_len=128, temperature=1.0, device="cpu"):
    model.eval()
    tokens = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
    env_t = torch.tensor([env], dtype=torch.float32, device=device)

    for _ in range(max_len - 1):
        logits, fn_logits = model(tokens, env_t)
        next_logits = logits[:, -1, :] / max(temperature, 1e-5)
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_id], dim=1)
        if next_id.item() == tokenizer.eos_id:
            break

    ids = tokens[0].tolist()
    seq = "".join(tokenizer.itos[i] for i in ids if i >= len(tokenizer.itos[:3]))
    predicted_function = int(fn_logits.argmax(-1).item())
    return seq, predicted_function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model_path", default="outputs/best_model.pt")
    parser.add_argument(
        "--env",
        default="0.7,0.5,0.6,0.8,0.4,0.3,0.9,0.4,0.2,0.5",
        help="Comma-separated environment vector with 10 dims",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = SequenceTokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EnvConditionedGenerator(
        vocab_size=tokenizer.vocab_size,
        env_dim=len(cfg["data"]["env_feature_columns"]),
        **cfg["model"],
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    env = [float(v) for v in args.env.split(",")]
    seq, fn = sample_sequence(model, tokenizer, env, max_len=cfg["data"]["max_sequence_len"], device=device)

    out = {"environment": env, "generated_sequence": seq, "predicted_function_class": fn}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
