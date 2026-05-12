import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
import argparse
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Import your models and dataset definitions
from model import ConditionalProteinGenerator, BaselineRNNGenerator, EnvDataset 
from model import VOCAB_SIZE, CHAR_TO_IDX
from train import INPUT_DIM, EMBED_SIZE, HIDDEN_SIZE

def evaluate_statistical_metrics(model, dataloader, criterion, device):
    """Calculates Loss, Perplexity, and Next-Token Accuracy on a dataset."""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    is_lstm = hasattr(model, 'env_to_cell')
    
    with torch.no_grad():
        for env_feat, seq in dataloader:
            env_feat, seq = env_feat.to(device), seq.to(device)
            
            # Recreate forward pass logic dynamically for LSTM vs RNN
            h0 = model.env_to_hidden(env_feat).unsqueeze(0)
            if is_lstm:
                c0 = model.env_to_cell(env_feat).unsqueeze(0)
                states = (h0, c0)
            else:
                states = h0
                
            embeds = model.embedding(seq[:, :-1])
            out, _ = model.lstm(embeds, states) if is_lstm else model.rnn(embeds, states)
            outputs = model.fc(out)
            
            # Reshape for loss
            outputs_flat = outputs.reshape(-1, VOCAB_SIZE)
            targets_flat = seq[:, 1:].reshape(-1)
            
            # Loss and Perplexity
            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.item()
            
            # Accuracy (Ignoring padding tokens)
            mask = targets_flat != CHAR_TO_IDX['<PAD>']
            preds = outputs_flat.argmax(dim=1)
            correct_predictions += (preds[mask] == targets_flat[mask]).sum().item()
            total_predictions += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    perplexity = np.exp(avg_loss)
    accuracy = correct_predictions / total_predictions * 100
    
    return avg_loss, perplexity, accuracy

def get_gravy_score(sequence):
    """Calculate the Grand Average of Hydropathy (GRAVY). High = hydrophobic, Low = hydrophilic."""
    # Filter out any padding or special tokens just in case
    clean_seq = "".join([aa for aa in sequence if aa in "ACDEFGHIKLMNPQRSTVWY"])
    if len(clean_seq) == 0: return 0
    try:
        analysis = ProteinAnalysis(clean_seq)
        return analysis.gravy()
    except:
        return 0

# ==========================================
# RUN EVALUATION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate conditional protein generators")
    parser.add_argument(
        "--data",
        default="data/cleaned_environmental_data_10K.csv",
        help="Path to the evaluation CSV file (default: data/cleaned_environmental_data_10K.csv)",
    )
    parser.add_argument(
        "--scaler",
        default="checkpoints/environmental_scaler_10K.pkl",
        help="Path to the fitted scaler pickle (default: checkpoints/environmental_scaler_10K.pkl)",
    )
    parser.add_argument(
        "--lstm-checkpoint",
        default="checkpoints/best_alien_protein_model_10K.pt",
        help="Path to the conditional LSTM checkpoint (default: checkpoints/best_alien_protein_model_10K.pt)",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default="checkpoints/best_baseline_alien_protein_model_10K.pt",
        help="Path to the baseline RNN checkpoint (default: checkpoints/best_baseline_alien_protein_model_10K.pt)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    # 1. Load Data for Validation
    df = pd.read_csv(args.data)
    
    # Re-calculate aggregates if necessary (like in your train script)
    tavg_cols = [f'tavg_{str(i).zfill(2)}' for i in range(1, 13)]
    prec_cols = [f'prec_{str(i).zfill(2)}' for i in range(1, 13)]
    srad_cols = [f'srad_{str(i).zfill(2)}' for i in range(1, 13)]
    df['Mean_Annual_Temp'] = df[tavg_cols].mean(axis=1)
    df['Total_Annual_Prec'] = df[prec_cols].sum(axis=1)
    df['Mean_Annual_Solar_Rad'] = df[srad_cols].mean(axis=1)
    df = df.dropna(subset=['Sequence', 'NASA_Temp_C', 'Total_Annual_Prec', 'Mean_Annual_Solar_Rad'])

    X = df[['NASA_Temp_C', 'Total_Annual_Prec', 'Mean_Annual_Solar_Rad']].values
    Y = df['Sequence'].tolist()

    # Load scaler and scale data
    scaler = joblib.load(args.scaler)
    X_scaled = scaler.transform(X)

    # We only care about the Validation set for accurate evaluation
    _, X_val, _, Y_val = train_test_split(X_scaled, Y, test_size=0.1, random_state=42)
    val_dataset = EnvDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss(ignore_index=CHAR_TO_IDX['<PAD>'])

    # 2. Initialize Models

    lstm_model = ConditionalProteinGenerator(INPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE).to(device)
    rnn_model = BaselineRNNGenerator(INPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE).to(device)

    # Load Weights (Change paths to wherever your actual checkpoints are saved)
    lstm_model.load_state_dict(torch.load(args.lstm_checkpoint, map_location=device, weights_only=True))
    rnn_model.load_state_dict(torch.load(args.baseline_checkpoint, map_location=device, weights_only=True))

    # 3. Statistical Evaluation
    print("\n" + "="*50)
    print("1. STATISTICAL METRICS (VALIDATION SET)")
    print("="*50)
    
    lstm_loss, lstm_perp, lstm_acc = evaluate_statistical_metrics(lstm_model, val_loader, criterion, device)
    print(f"[LSTM] Loss: {lstm_loss:.4f} | Perplexity: {lstm_perp:.2f} | Next-Token Acc: {lstm_acc:.2f}%")

    rnn_loss, rnn_perp, rnn_acc = evaluate_statistical_metrics(rnn_model, val_loader, criterion, device)
    print(f"[RNN]  Loss: {rnn_loss:.4f} | Perplexity: {rnn_perp:.2f} | Next-Token Acc: {rnn_acc:.2f}%")

    # 4. Biological Evaluation (Inference)
    print("\n" + "="*50)
    print("2. BIOLOGICAL METRICS (GENERATED SEQUENCES)")
    print("="*50)
    
    # Import the updated generate_protein function from train.py (or paste it above in this script)
    from train import generate_protein
    from train_baseline import generate_protein as generate_protein_baseline
    
    # Test a Hot Environment (Thermophile) vs Cold Environment (Psychrophile)
    env_hot = {"temp_c": 85.0, "precip": 10.0, "radiation": 35000.0}
    env_cold = {"temp_c": -15.0, "precip": 500.0, "radiation": 5000.0}

    print("--- Hot Environment (85°C) ---")
    lstm_hot_seq = generate_protein(
        lstm_model,
        scaler,
        temp_c=env_hot["temp_c"],
        precip=env_hot["precip"],
        radiation=env_hot["radiation"],
        device=device,
    )
    rnn_hot_seq = generate_protein_baseline(
        rnn_model,
        scaler,
        temp_c=env_hot["temp_c"],
        precip=env_hot["precip"],
        radiation=env_hot["radiation"],
        device=device,
    )
    
    print(f"[LSTM] Length: {len(lstm_hot_seq):03d} | GRAVY Score: {get_gravy_score(lstm_hot_seq):.3f}")
    print(f"[RNN]  Length: {len(rnn_hot_seq):03d} | GRAVY Score: {get_gravy_score(rnn_hot_seq):.3f}")

    print("\n--- Cold Environment (-15°C) ---")
    lstm_cold_seq = generate_protein(
        lstm_model,
        scaler,
        temp_c=env_cold["temp_c"],
        precip=env_cold["precip"],
        radiation=env_cold["radiation"],
        device=device,
    )
    rnn_cold_seq = generate_protein_baseline(
        rnn_model,
        scaler,
        temp_c=env_cold["temp_c"],
        precip=env_cold["precip"],
        radiation=env_cold["radiation"],
        device=device,
    )
    
    print(f"[LSTM] Length: {len(lstm_cold_seq):03d} | GRAVY Score: {get_gravy_score(lstm_cold_seq):.3f}")
    print(f"[RNN]  Length: {len(rnn_cold_seq):03d} | GRAVY Score: {get_gravy_score(rnn_cold_seq):.3f}")
    print("==================================================")