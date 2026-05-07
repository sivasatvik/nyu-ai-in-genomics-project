from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import EnvSequenceDataset, SequenceTokenizer
from models.conditional_generator import EnvConditionedGenerator
from utils.io import ensure_dirs, load_config, save_json, set_seed


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def run_epoch(model, loader, optimizer, device, pad_id, fn_weight=0.3):
    ce = nn.CrossEntropyLoss(ignore_index=pad_id)
    fn_loss_fn = nn.CrossEntropyLoss()
    train_mode = optimizer is not None
    model.train(train_mode)

    total = {"loss": 0.0, "lm": 0.0, "fn": 0.0}
    correct, denom = 0, 0
    for batch in tqdm(loader, leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        env = batch["env"].to(device)
        fn_label = batch["fn_label"].to(device)

        logits, fn_logits = model(x, env)
        lm_loss = ce(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        fn_loss = fn_loss_fn(fn_logits, fn_label)
        loss = lm_loss + fn_weight * fn_loss

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total["loss"] += loss.item()
        total["lm"] += lm_loss.item()
        total["fn"] += fn_loss.item()
        preds = fn_logits.argmax(-1)
        correct += (preds == fn_label).sum().item()
        denom += fn_label.size(0)

    n = max(1, len(loader))
    return {
        "loss": total["loss"] / n,
        "lm_loss": total["lm"] / n,
        "fn_loss": total["fn"] / n,
        "fn_acc": correct / max(1, denom),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", default="data/processed/dataset.jsonl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    ensure_dirs(cfg["paths"]["output_dir"])

    rows = load_jsonl(args.dataset)
    train_rows, temp_rows = train_test_split(rows, test_size=0.2, random_state=cfg["seed"])
    val_rows, test_rows = train_test_split(temp_rows, test_size=0.5, random_state=cfg["seed"])

    tokenizer = SequenceTokenizer()
    max_len = cfg["data"]["max_sequence_len"]
    train_ds = EnvSequenceDataset(train_rows, tokenizer, max_len)
    val_ds = EnvSequenceDataset(val_rows, tokenizer, max_len)
    test_ds = EnvSequenceDataset(test_rows, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EnvConditionedGenerator(
        vocab_size=tokenizer.vocab_size,
        env_dim=len(cfg["data"]["env_feature_columns"]),
        **cfg["model"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )

    history = {"train": [], "val": []}
    best_val = np.inf
    out_dir = Path(cfg["paths"]["output_dir"])
    best_path = out_dir / "best_model.pt"

    for epoch in range(cfg["training"]["num_epochs"]):
        tr = run_epoch(model, train_loader, optimizer, device, tokenizer.pad_id)
        with torch.no_grad():
            va = run_epoch(model, val_loader, None, device, tokenizer.pad_id)
        history["train"].append(tr)
        history["val"].append(va)
        print(f"epoch={epoch+1} train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} val_fn_acc={va['fn_acc']:.3f}")

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    with torch.no_grad():
        te = run_epoch(model, test_loader, None, device, tokenizer.pad_id)

    metrics = {"best_val_loss": best_val, "test": te, "device": device}
    save_json(history, str(out_dir / "history.json"))
    save_json(metrics, str(out_dir / "metrics.json"))
    print("[ok] Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
