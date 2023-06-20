import numpy as np
import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(Transformer, self).__init__()
        # This is our holy embedding layer - the topic of this post
        self.embedding = nn.Embedding(vocab_size, d_model)

        # This is a transformer layer. It contains encoder and decoder
        self.transformer = nn.Transformer(d_model, nhead, num_layers)

        #This is the final fully connected layer that predicts the probability of each word
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Pass input through the embedding layer
        x = self.embedding(x)

        # Pass input through the transformer layers (NOTE: This input is usually concatenated with positional encoding. I left it out for simplicity)
        x = self.transformer(x)
        # Pass input through the final linear layer
        x = self.fc(x)
        return x

# Initialize the model
vocab_size = 10
d_model = 50
nhead = 2
num_layers = 3
model = Transformer(vocab_size, d_model, nhead, num_layers)