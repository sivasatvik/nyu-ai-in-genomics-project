from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class EnvConditionedGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        env_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
        function_classes: int,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.env_proj = nn.Linear(env_dim, d_model)
        self.pos = PositionalEncoding(d_model=d_model, max_len=max_len)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.function_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, function_classes)
        )

    def forward(self, tokens: torch.Tensor, env: torch.Tensor):
        tok = self.token_emb(tokens)
        env_tok = self.env_proj(env).unsqueeze(1)
        h = tok + env_tok
        h = self.pos(h)
        h = self.decoder(h)

        logits = self.lm_head(h)
        pooled = h.mean(dim=1)
        fn_logits = self.function_head(pooled)
        return logits, fn_logits
