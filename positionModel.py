import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from positionalEncoding import PositionalEncoding


class PositionNN(nn.Module):

    def __init__(
        self,
        num_tokens,
        dim_model,
        dropout_p=0.1,
    ):
        super().__init__()
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding( batch_size=64,
            dim_model=dim_model, dropout_p=dropout_p, max_len=46
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
      
        self.out =  nn.Sequential(
            nn.Linear(46* dim_model,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64,2)
        )
        
    def forward(self, src,  ):

        src = self.embedding(src) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)

        #src = src.permute(1,0,2)

        return self.out(src.view(src.shape[0],-1))
    
      


