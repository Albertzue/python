import torch
import numpy as np
import torch.nn as nn

embedding = nn.Embedding(4,3)

input = torch.LongTensor([[1, 2, 0, 3,3,], [2, 3, 2,1,1 ]])
print(embedding(input))