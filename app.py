from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import torch

from src.data.dataset import SequenceTokenizer
from src.models.conditional_generator import EnvConditionedGenerator
from src.utils.io import load_config


@st.cache_resource
def load_inference_artifacts(config_path: str, model_path: str):
    cfg = load_config(config_path)
    tokenizer = SequenceTokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EnvConditionedGenerator(
        vocab_size=tokenizer.vocab_size,
        env_dim=len(cfg["data"]["env_feature_columns"]),
        **cfg["model"],
    ).to(device)

    model_loaded = False
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        model_loaded = True
    model.eval()
    return cfg, tokenizer, model, device, model_loaded


def apply_top_k(logits: torch.Tensor, k: int | None):
    if k is None or k <= 0 or k >= logits.shape[-1]:
        return logits
    top_k_logits, _ = torch.topk(logits, k, dim=-1)
    min_val = top_k_logits[..., -1:].clone()
    logits_filtered = logits.clone()
    logits_filtered[logits_filtered < min_val] = float("-inf")
    return logits_filtered


def apply_top_p(probs: torch.Tensor, p: float | None):
    if p is None or p <= 0.0 or p >= 1.0:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumsum > p
    sorted_indices_to_remove[..., 0] = False

    probs_filtered = probs.clone()
    probs_filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_probs * (~sorted_indices_to_remove).to(sorted_probs.dtype))
    denom = probs_filtered.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return probs_filtered / denom


def sample_sequence(
    model,
    tokenizer,
    env,
    min_len: int,
    max_len: int,
    device: str,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
):
    tokens = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
    env_t = torch.tensor([env], dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(max_len - 1):
            logits, fn_logits = model(tokens, env_t)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            next_logits = apply_top_k(next_logits, top_k)
            if tokens.size(1) - 1 < min_len:
                next_logits[:, tokenizer.eos_id] = float("-inf")
            probs = torch.softmax(next_logits, dim=-1)
            probs = apply_top_p(probs, top_p)
            next_id = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_id], dim=1)
            if next_id.item() == tokenizer.eos_id:
                break

    ids = tokens[0].tolist()
    seq = "".join(tokenizer.itos[i] for i in ids if i >= 3)
    fn_class = int(fn_logits.argmax(-1).item())
    return seq, fn_class


def main():
    st.set_page_config(page_title="Alien Environment Protein Generator", layout="wide")
    st.title("Alien Environment → Amino Acid Sequence (Base UI)")
    st.caption("Adjust environment conditions and generate a protein sequence from the saved model.")

    config_path = st.sidebar.text_input("Config path", value="config.yaml")
    model_path = st.sidebar.text_input("Model checkpoint", value="outputs/best_model.pt")
    min_len = st.sidebar.number_input(
        "Minimum generated length", min_value=1, max_value=5000, value=80, step=1
    )
    temperature = st.sidebar.slider("Sampling temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    top_k = st.sidebar.number_input(
        "Top-k filtering (0 disables)", min_value=0, max_value=5000, value=0, step=1
    )
    top_p = st.sidebar.slider(
        "Top-p / nucleus filtering (1.0 disables)", min_value=0.0, max_value=1.0, value=1.0, step=0.01
    )

    cfg, tokenizer, model, device, model_loaded = load_inference_artifacts(config_path, model_path)
    st.sidebar.write(f"Device: `{device}`")

    if not model_loaded:
        st.sidebar.error(f"Model checkpoint not found: {model_path}")
        st.error("Checkpoint missing. Train the model first or provide a valid checkpoint path.")
        st.stop()

    st.subheader("Environment Conditions")
    env_features = cfg["data"]["env_feature_columns"]
    env_values = []

    cols = st.columns(2)
    for i, feat in enumerate(env_features):
        with cols[i % 2]:
            env_values.append(
                st.slider(feat, min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=f"env_{feat}")
            )

    generate = st.button("Generate sequence")
    if generate:
        seq, fn_class = sample_sequence(
            model=model,
            tokenizer=tokenizer,
            env=env_values,
            min_len=int(min_len),
            max_len=cfg["data"]["max_sequence_len"],
            device=device,
            temperature=temperature,
            top_k=int(top_k) if top_k > 0 else None,
            top_p=top_p if top_p < 1.0 else None,
        )
        st.success("Sequence generated")
        st.write("**Predicted function class:**", fn_class)
        st.code(seq, language="text")
        st.json(
            {
                "env_vector": env_values,
                "predicted_function_class": fn_class,
                "sequence_length": len(seq),
                "sampling": {
                    "min_len": int(min_len),
                    "temperature": temperature,
                    "top_k": int(top_k) if top_k > 0 else None,
                    "top_p": top_p if top_p < 1.0 else None,
                },
            }
        )

    st.markdown("---")
    st.write("Tip: Train first (`python src/train.py ...`) so `outputs/best_model.pt` exists.")


if __name__ == "__main__":
    main()
