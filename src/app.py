if __package__ is None or __package__ == "":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
from model import ConditionalProteinGenerator, VOCAB_SIZE
from train import INPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, generate_protein

# ==========================================
# 2. CACHED MODEL LOADING
# ==========================================
@st.cache_resource
def load_system(checkpoint_path, scaler_path):
    """Loads the model and scaler once and caches them in memory for fast UI response."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the Model
    model = ConditionalProteinGenerator(INPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE).to(device)
    # map_location ensures it loads on CPU if you trained on GPU but are running the UI on a laptop
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    # 2. Load the pre-fitted Scaler (Instant and 100% accurate!)
    scaler = joblib.load(scaler_path)
    
    return model, scaler, device


# ==========================================
# 4. STREAMLIT UI DESIGN
# ==========================================
st.set_page_config(page_title="Alien Genomics AI", page_icon="🧬", layout="wide")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--checkpoint",
    default="checkpoints/best_alien_protein_model.pt",
    help="Path to the trained checkpoint file",
)
parser.add_argument(
    "--scaler",
    default="checkpoints/environmental_scaler.pkl",
    help="Path to the fitted scaler pickle file",
)
args, _ = parser.parse_known_args()

model, scaler, device = load_system(args.checkpoint, args.scaler)

st.title("🧬 Alien Protein Generator")
st.markdown("""
Adjust the environmental parameters on the left to simulate different planetary conditions. 
The AI will generate a novel amino acid sequence structurally adapted to survive those exact environmental factors.
""")

# Layout: Sidebar for controls, Main area for results
st.sidebar.header("🪐 Planetary Environment")
st.sidebar.markdown("Use the knobs to adjust the climate.")

# The "Knobs" (Sliders)
temp_input = st.sidebar.slider("Surface Temperature (°C)", min_value=-50.0, max_value=150.0, value=25.0, step=1.0)
precip_input = st.sidebar.slider("Annual Precipitation (mm)", min_value=0.0, max_value=5000.0, value=1000.0, step=10.0)
rad_input = st.sidebar.slider("Solar Radiation (kJ m-2 day-1)", min_value=0.0, max_value=40000.0, value=15000.0, step=100.0)

# Preset buttons for fun
st.sidebar.markdown("---")
st.sidebar.markdown("**Presets:**")
col1, col2 = st.sidebar.columns(2)
if col1.button("🌋 Lava Planet"):
    temp_input, precip_input, rad_input = 120.0, 0.0, 35000.0
if col2.button("🧊 Ice Moon"):
    temp_input, precip_input, rad_input = -40.0, 50.0, 2000.0

# Main generation area
st.subheader("Generated Sequence")

if st.button("🚀 Generate Protein", type="primary"):
    with st.spinner("Synthesizing novel sequence..."):
        generated_seq = generate_protein(
            model,
            scaler,
            temp_input,
            precip_input,
            rad_input,
            device,
        )
        
        if len(generated_seq) > 0:
            st.success("Synthesis Complete!")
            # Display sequence in a monospace code block for biological accuracy
            st.code(generated_seq, language="text")
            st.markdown(f"**Sequence Length:** {len(generated_seq)} Amino Acids")
        else:
            st.warning("The model generated an empty sequence. Try adjusting the parameters.")