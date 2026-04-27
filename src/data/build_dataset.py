from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))


def random_sequence(length: int, env_vec: np.ndarray) -> str:
    probs = np.ones(len(AA), dtype=float)
    sulfur = float(env_vec[3]) if len(env_vec) > 3 else 0.1
    iron = float(env_vec[6]) if len(env_vec) > 6 else 0.1
    bias = min(3.0, 1.0 + sulfur + iron)
    for aa in ["C", "M", "H"]:
        probs[np.where(AA == aa)[0][0]] *= bias
    probs /= probs.sum()
    return "".join(np.random.choice(AA, size=length, p=probs))


def construct_env_vector(row: pd.Series) -> list[float]:
    base = np.clip(np.random.normal(0.5, 0.2, size=10), 0.0, 1.0)
    if "st_met" in row and pd.notna(row.get("st_met")):
        base[6] = float(np.clip((row["st_met"] + 1.0) / 2.0, 0.0, 1.0))
    if "st_teff" in row and pd.notna(row.get("st_teff")):
        base[2] = float(np.clip((row["st_teff"] - 2500) / 7000, 0.0, 1.0))
    return base.tolist()


def load_extremophile_records(raw_dir: Path, holdout_fraction: float) -> tuple[list[dict], list[dict]]:
    path = raw_dir / "extremophile_sequences.jsonl"
    if not path.exists():
        return [], []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            rows.append(
                {
                    "id": f"ext_{i}",
                    "env_vector": r["env_vector"],
                    "sequence": r["sequence"],
                    "function_label": int(r.get("function_label", 0)),
                    "source": "extremophile",
                    "habitat": r.get("habitat", "unknown"),
                    "organism_name": r.get("organism_name", "unknown"),
                }
            )

    if not rows:
        return [], []

    n_holdout = max(1, int(len(rows) * holdout_fraction))
    np.random.shuffle(rows)
    return rows[n_holdout:], rows[:n_holdout]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_path", default="data/processed/dataset.jsonl")
    parser.add_argument("--extreme_eval_path", default="data/processed/extremophile_eval.jsonl")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--extreme_holdout_fraction", type=float, default=0.2)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out_path)
    eval_path = Path(args.extreme_eval_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exo_path = raw_dir / "nasa_exoplanets.csv"
    exo_df = pd.read_csv(exo_path) if exo_path.exists() else pd.DataFrame([{} for _ in range(args.n_samples)])

    records = []
    for i in range(args.n_samples):
        row = exo_df.iloc[i % len(exo_df)] if len(exo_df) > 0 else pd.Series(dtype=float)
        env = np.array(construct_env_vector(row))
        seq_len = int(np.random.randint(80, 220))
        seq = random_sequence(seq_len, env)
        fn_label = int(np.digitize(env.mean(), bins=np.linspace(0.2, 0.9, 7)))
        records.append(
            {
                "id": f"sim_{i}",
                "env_vector": env.tolist(),
                "sequence": seq,
                "function_label": min(fn_label, 7),
                "source": "simulated",
            }
        )

    ext_train, ext_eval = load_extremophile_records(raw_dir, args.extreme_holdout_fraction)
    records.extend(ext_train)

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    with eval_path.open("w", encoding="utf-8") as f:
        for r in ext_eval:
            f.write(json.dumps(r) + "\n")

    print(f"[ok] wrote {len(records)} training records to {out_path}")
    print(f"[ok] wrote {len(ext_eval)} extremophile eval records to {eval_path}")


if __name__ == "__main__":
    main()
