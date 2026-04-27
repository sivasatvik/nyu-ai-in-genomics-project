from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import EnvSequenceDataset, SequenceTokenizer
from models.conditional_generator import EnvConditionedGenerator
from utils.io import load_config


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def evaluate_on_extremophiles(model, loader, pad_id: int, device: str):
    ce = nn.CrossEntropyLoss(ignore_index=pad_id)
    clf = nn.CrossEntropyLoss()

    model.eval()
    lm_losses, fn_losses = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            env = batch["env"].to(device)
            fn_label = batch["fn_label"].to(device)

            logits, fn_logits = model(x, env)
            lm_losses.append(ce(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
            fn_losses.append(clf(fn_logits, fn_label).item())
            pred = fn_logits.argmax(-1)
            correct += (pred == fn_label).sum().item()
            total += fn_label.size(0)

    return {
        "extremophile_lm_loss": float(sum(lm_losses) / max(1, len(lm_losses))),
        "extremophile_function_loss": float(sum(fn_losses) / max(1, len(fn_losses))),
        "extremophile_function_acc": float(correct / max(1, total)),
        "n_eval_samples": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model_path", default="outputs/best_model.pt")
    parser.add_argument("--extreme_eval_path", default="data/processed/extremophile_eval.jsonl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = load_jsonl(args.extreme_eval_path)
    if not rows:
        raise ValueError("No extremophile evaluation rows found. Run data download/build steps first.")

    tokenizer = SequenceTokenizer()
    ds = EnvSequenceDataset(rows, tokenizer, cfg["data"]["max_sequence_len"])
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EnvConditionedGenerator(
        vocab_size=tokenizer.vocab_size,
        env_dim=len(cfg["data"]["env_feature_columns"]),
        **cfg["model"],
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    metrics = evaluate_on_extremophiles(model, loader, tokenizer.pad_id, device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
