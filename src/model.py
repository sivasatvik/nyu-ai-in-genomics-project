import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

if __package__ is None or __package__ == "":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

# ==========================================
# 1. DATA PREPROCESSING & TOKENIZATION
# ==========================================
# Standard 20 amino acids + special tokens
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
VOCAB = ['<PAD>', '<SOS>', '<EOS>'] + AMINO_ACIDS
CHAR_TO_IDX = {ch: i for i, ch in enumerate(VOCAB)}
IDX_TO_CHAR = {i: ch for i, ch in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

class EnvDataset(Dataset):
    def __init__(self, features, sequences, max_len=200):
        self.features = torch.FloatTensor(features)
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences.iloc[idx]
        # Truncate and add Start-Of-Sequence <SOS> and End-Of-Sequence <EOS>
        seq = seq[:self.max_len-2] 
        tokens = ['<SOS>'] + list(seq) + ['<EOS>']
        
        # Convert characters to indices, pad with <PAD> (0) up to max_len
        token_indices = [CHAR_TO_IDX.get(char, CHAR_TO_IDX['<PAD>']) for char in tokens]
        padding = [CHAR_TO_IDX['<PAD>']] * (self.max_len - len(token_indices))
        token_indices.extend(padding)
        
        return self.features[idx], torch.LongTensor(token_indices)


# ==========================================
# 2. THE CONDITIONAL GENERATOR MODEL
# ==========================================
class ConditionalProteinGenerator(nn.Module):
    def __init__(self, input_feature_dim, embed_size, hidden_size, vocab_size):
        super(ConditionalProteinGenerator, self).__init__()
        self.hidden_size = hidden_size
        
        # Maps environmental variables to the LSTM's initial hidden state
        self.env_to_hidden = nn.Linear(input_feature_dim, hidden_size)
        self.env_to_cell = nn.Linear(input_feature_dim, hidden_size)
        
        # Standard Sequence components
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, env_features, sequences):
        # 1. Process environment into initial LSTM states
        h0 = self.env_to_hidden(env_features).unsqueeze(0) # Shape: (1, batch, hidden)
        c0 = self.env_to_cell(env_features).unsqueeze(0)
        
        # 2. Embed the protein sequence
        # We don't pass the last character into the LSTM because it has nothing to predict
        embeds = self.embedding(sequences[:, :-1]) 
        
        # 3. Pass through LSTM
        lstm_out, _ = self.lstm(embeds, (h0, c0))
        
        # 4. Predict next amino acid at each step
        predictions = self.fc(lstm_out)
        return predictions