import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
from positionalEncoding import PositionalEncoding

num_features = 23
dropout_p = 0.1
num_encoder_layers = num_decoder_layers = 6
n_head = 8
position_max_len=100
class Transformer(nn.Module):
    def __init__(self, num_tokens, dim_model,num_heads, num_encoder_layers, num_decoder_layers,position_max_len,dropout_p):
        super(Transformer, self).__init__()
        
        self.model_type ="Transformer"
        self.dim_model = dim_model
        
        self.positional_encoder = PositionalEncoding(
            dim_model = dim_model, dropout_p = dropout_p,  max_len = position_max_len
        )
        
        self.embedding = nn.Embedding(num_tokens, dim_model)
        
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers= num_decoder_layers,
            dropout = dropout_p
        )
        
        self.out = nn.Linear(dim_model,num_tokens)
        
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask = None, tgt_pad_mask = None):
        src = self.embedding(src)* math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)
        
        transformer_out = self.transformer(
            src,tgt,tgt_mask = tgt_mask,
            src_key_padding_mask = src_pad_mask, 
            tgt_key_padding_mask=tgt_pad_mask)
        
        out = self.out(transformer_out)
        
        return out
    
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)