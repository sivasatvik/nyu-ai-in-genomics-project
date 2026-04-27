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

    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return cfg, tokenizer, model, device


def sample_sequence(model, tokenizer, env, max_len: int, device: str, temperature: float):
    tokens = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
    env_t = torch.tensor([env], dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(max_len - 1):
            logits, fn_logits = model(tokens, env_t)
            probs = torch.softmax(logits[:, -1, :] / max(temperature, 1e-5), dim=-1)
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
    temperature = st.sidebar.slider("Sampling temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

    cfg, tokenizer, model, device = load_inference_artifacts(config_path, model_path)
    st.sidebar.write(f"Device: `{device}`")

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
            max_len=cfg["data"]["max_sequence_len"],
            device=device,
            temperature=temperature,
        )
        st.success("Sequence generated")
        st.write("**Predicted function class:**", fn_class)
        st.code(seq, language="text")
        st.json({"env_vector": env_values, "predicted_function_class": fn_class, "sequence_length": len(seq)})

    st.markdown("---")
    st.write("Tip: Train first (`python src/train.py ...`) so `outputs/best_model.pt` exists.")


if __name__ == "__main__":
    main()
