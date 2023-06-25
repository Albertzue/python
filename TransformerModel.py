import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from positionalEncoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(
        self,
        nhead=8,
        dim_feedforward=128,
        num_layers=6,
        dropout=0.1,
    ):

        super().__init__()

        vocab_size, d_model = 4,46
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        self.emb = nn.Embedding(vocab_size,d_model)

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            vocab_size=vocab_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Sequential(
                nn.Linear(d_model,dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64,2)
            )
        self.d_model = d_model

    def forward(self, x):
        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)

        return x