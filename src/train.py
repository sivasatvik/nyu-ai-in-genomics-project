if __package__ is None or __package__ == "":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import train_test_split
from model import ConditionalProteinGenerator, EnvDataset, VOCAB_SIZE, CHAR_TO_IDX, IDX_TO_CHAR
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
import os
import argparse

feature_cols = ['NASA_Temp_C', 'Total_Annual_Prec', 'Mean_Annual_Solar_Rad']

# Initialize model
INPUT_DIM = len(feature_cols)
EMBED_SIZE = 64
HIDDEN_SIZE = 256


def generate_protein(model, scaler, temp_c, precip, radiation, device, max_len=100):
    """Generate a protein sequence given environmental conditions."""
    model.eval()
    with torch.no_grad():
        # Prepare the environmental "prompt"
        env_raw = np.array([[temp_c, precip, radiation]])
        env_scaled = scaler.transform(env_raw)
        env_tensor = torch.FloatTensor(env_scaled).to(device)
        
        # Get initial LSTM states from environment
        h0 = model.env_to_hidden(env_tensor).unsqueeze(0)
        c0 = model.env_to_cell(env_tensor).unsqueeze(0)
        
        # Start the sequence with <SOS>
        current_token = torch.LongTensor([[CHAR_TO_IDX['<SOS>']]]).to(device)
        generated_sequence = []
        
        # Generate amino acids one by one
        for _ in range(max_len):
            embedded = model.embedding(current_token)
            lstm_out, (h0, c0) = model.lstm(embedded, (h0, c0))
            prediction = model.fc(lstm_out)
            
            # Get the most likely next amino acid
            next_token_idx = prediction.argmax(dim=2).item()
            
            # Stop if the model decides the protein is finished
            if next_token_idx == CHAR_TO_IDX['<EOS>']:
                break
                
            generated_sequence.append(IDX_TO_CHAR[next_token_idx])
            
            # Feed the generated amino acid back in for the next step
            current_token = torch.LongTensor([[next_token_idx]]).to(device)
            
        return "".join(generated_sequence)


def save_loss_plot(train_losses, val_losses, plot_path):
    """Save a training/validation loss curve to disk."""
    if not plot_path:
        return

    plot_dir = os.path.dirname(plot_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved loss plot to {plot_path}")


def main(csv_path, scaler_path, checkpoint_path, loss_plot_path="figures/train_loss_curve.png"):
    """Main training and inference pipeline."""
    # Load data and prepare features
    df = pd.read_csv(csv_path)

    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Calculate Annual Aggregates from monthly columns
    tavg_cols = [f'tavg_{str(i).zfill(2)}' for i in range(1, 13)]
    prec_cols = [f'prec_{str(i).zfill(2)}' for i in range(1, 13)]
    srad_cols = [f'srad_{str(i).zfill(2)}' for i in range(1, 13)]

    df['Mean_Annual_Temp'] = df[tavg_cols].mean(axis=1)
    df['Total_Annual_Prec'] = df[prec_cols].sum(axis=1)
    df['Mean_Annual_Solar_Rad'] = df[srad_cols].mean(axis=1)

    # Drop rows with missing target sequences or features
    df = df.dropna(subset=['Sequence', 'NASA_Temp_C', 'Total_Annual_Prec', 'Mean_Annual_Solar_Rad'])

    X = df[feature_cols].values
    Y = df['Sequence']

    # Scale features (Crucial for Neural Networks)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Save the scaler for later use in the app
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")

    # Train/Test Split
    X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y, test_size=0.1, random_state=42)

    train_dataset = EnvDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = EnvDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalProteinGenerator(INPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE).to(device)

    # ==========================================
    # 1. TRAINING LOOP
    # ==========================================
    criterion = nn.CrossEntropyLoss(ignore_index=CHAR_TO_IDX['<PAD>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    EPOCHS = 100 # Increase to 50-100 for actual training

    best_val_loss = float('inf') # Keep track of the best loss
    train_losses = []
    val_losses = []

    print(f"Training on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        for env_feat, seq in train_loader:
            env_feat, seq = env_feat.to(device), seq.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(env_feat, seq)
            
            # Reshape for loss calculation: 
            # Loss expects (batch_size * seq_len, vocab_size) and targets (batch_size * seq_len)
            # We shift targets by 1 because output[0] predicts seq[1]
            outputs = outputs.reshape(-1, VOCAB_SIZE)
            targets = seq[:, 1:].reshape(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for env_feat, seq in val_loader:
                env_feat, seq = env_feat.to(device), seq.to(device)
                outputs = model(env_feat, seq)
                outputs = outputs.reshape(-1, VOCAB_SIZE)
                targets = seq[:, 1:].reshape(-1)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # --- CHECKPOINTING LOGIC ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the model's exact weights at this state
            torch.save(model.state_dict(), checkpoint_path)
            status = "=> Saved New Best Model!"
        else:
            status = ""
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} {status}")

    save_loss_plot(train_losses, val_losses, loss_plot_path)

    # ==========================================
    # 2. INFERENCE: LOADING CHECKPOINT AND TESTING
    # ==========================================
    def inference_single_env():
        """Test the model with Extreme Exoplanet Parameters."""
        # Let's simulate a hyper-arid, high-radiation, high-heat planet
        alien_temp = 85.0    # 85°C 
        alien_precip = 10.0  # Basically no rain
        alien_rad = 35000.0  # Intense solar radiation

        novel_protein = generate_protein(model, scaler, alien_temp, alien_precip, alien_rad, device)

        print("\n" + "="*50)
        print(f"Generated Protein for 85°C, High Radiation Environment:")
        print(novel_protein)
        print("="*50)

    def inference_multiple_envs():
        """Test with multiple environment scenarios."""
        print("\n" + "="*50)
        print("Loading Best Checkpoint for Inference...")
        # 1. Initialize a blank model architecture
        best_model = ConditionalProteinGenerator(INPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE).to(device)
        # 2. Load the saved weights into it
        best_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        best_model.eval()

        # 3. Define various alien environments to test!
        environments = [
            {"name": "Earth-like (Temperate)", "temp": 15.0, "precip": 1000.0, "rad": 15000.0},
            {"name": "Scorched Desert Planet", "temp": 85.0, "precip": 10.0, "rad": 35000.0},
            {"name": "Frozen Ice Moon", "temp": -40.0, "precip": 50.0, "rad": 2000.0},
        ]

        for env in environments:
            seq = generate_protein(
                best_model, 
                scaler, 
                temp_c=env["temp"], 
                precip=env["precip"], 
                radiation=env["rad"],
                device=device
            )
            print(f"\nEnvironment: {env['name']}")
            print(f"Inputs -> Temp: {env['temp']}°C | Precip: {env['precip']}mm | Rad: {env['rad']}")
            print(f"Generated Sequence: {seq}")
        print("="*50)

    # Run inference tests
    inference_single_env()
    inference_multiple_envs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train conditional protein generator from environmental CSV")
    parser.add_argument(
        "-d",
        "--data",
        default="data/cleaned_environmental_data_10K.csv",
        help="Path to input CSV file (default: data/cleaned_environmental_data_10K.csv)",
    )
    parser.add_argument(
        "-s",
        "--scaler",
        default="checkpoints/environmental_scaler_10K.pkl",
        help="Path to scaler pickle file (default: checkpoints/environmental_scaler_10K.pkl)",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="checkpoints/best_alien_protein_model_10K.pt",
        help="Path to model checkpoint file (default: checkpoints/best_alien_protein_model_10K.pt)",
    )
    parser.add_argument(
        "-l",
        "--loss-plot",
        default="figures/train_loss_curve.png",
        help="Path to save the training loss plot (default: figures/train_loss_curve.png)",
    )
    args = parser.parse_args()
    main(args.data, args.scaler, args.checkpoint, args.loss_plot)