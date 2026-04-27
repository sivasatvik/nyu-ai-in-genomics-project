from __future__ import annotations

import argparse
import json
import torch

from data.dataset import SequenceTokenizer
from models.conditional_generator import EnvConditionedGenerator
from utils.io import load_config


def apply_top_k(logits, k=50):
    """Keep only top-k logits, set others to -inf."""
    if k >= logits.shape[-1]:
        return logits
    top_k_logits, _ = torch.topk(logits, k, dim=-1)
    min_val = top_k_logits[..., -1:].clone()
    logits_filtered = logits.clone()
    logits_filtered[logits_filtered < min_val] = float('-inf')
    return logits_filtered


def apply_top_p(probs, p=0.9):
    """Keep tokens until cumulative probability >= p (nucleus sampling)."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumsum > p
    sorted_indices_to_remove[..., 0] = False  # keep at least one token
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs_filtered = probs.clone()
    probs_filtered[indices_to_remove] = 0.0
    return probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)


def sample_sequence(model, tokenizer, env, max_len=128, temperature=1.0, top_k=None, top_p=None, device="cpu"):
    """
    Generate amino acid sequence with optional top-k or top-p sampling.
    Args:
        model: EnvConditionedGenerator model
        tokenizer: SequenceTokenizer instance
        env: environment vector
        max_len: maximum sequence length
        temperature: softmax temperature (higher = more diverse)
        top_k: keep only top-k tokens (None = disabled)
        top_p: nucleus sampling threshold (None = disabled)
        device: device to run model on
    """
    model.eval()
    tokens = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
    env_t = torch.tensor([env], dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(max_len - 1):
            logits, fn_logits = model(tokens, env_t)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            # Apply top-k filtering if specified
            if top_k is not None and top_k > 0:
                next_logits = apply_top_k(next_logits, k=top_k)
            
            probs = torch.softmax(next_logits, dim=-1)
            
            # Apply top-p filtering if specified
            if top_p is not None and 0 < top_p < 1.0:
                probs = apply_top_p(probs, p=top_p)
            
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
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for sampling")
    parser.add_argument("--top_k", type=int, default=None, help="Keep only top-k tokens (None to disable)")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling threshold (None to disable)")
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
    seq, fn = sample_sequence(
        model, tokenizer, env, 
        max_len=cfg["data"]["max_sequence_len"], 
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device
    )

    out = {
        "environment": env, 
        "generated_sequence": seq, 
        "predicted_function_class": fn,
        "sampling_config": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p
        }
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
