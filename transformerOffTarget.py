import torch
import torch.nn as nn
import torch.optim as optim
from trasnformerDataset import OfftargetDataset_train
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import torch.nn.functional as F
from TransformerModel import Transformer 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
ntokens = 46
emsize = 46
nhead = 2
nhid = 512
nlayers = 3
dropout  = 0.1
learning_rate = 1e-2
generator = torch.Generator().manual_seed(42)
dataset = OfftargetDataset_train()
train_set, val_set = torch.utils.data.random_split(dataset,[0.8,0.2], generator= generator)

train_dataloader = DataLoader(batch_size=batch_size, shuffle=True, dataset=train_set)
val_dataloader = DataLoader(batch_size=batch_size, shuffle=False, dataset=val_set)




loss_fn = nn.CrossEntropyLoss()



epochs = 10
model = Transformer(nhead,dim_feedforward=64,num_layers= nlayers,dropout=0.1).to(device)

criterion = nn.CrossEntropyLoss()

lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

torch.manual_seed(0)

print("starting")
for epoch in range(epochs):
    print(f"{epoch=}")
    epoch_loss = 0
    epoch_correct = 0
    epoch_count = 0
    for seqs, labels in train_dataloader:
        predictions = model(seqs.view(seqs.shape[0],-1))

        loss = criterion(predictions, labels)

        correct = predictions.argmax(axis=1) == labels
        acc = correct.sum().item() / correct.size(0)

        epoch_correct += correct.sum().item()
        epoch_count += correct.size(0)

        epoch_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()

    with torch.no_grad():
            test_epoch_loss = 0
            test_epoch_correct = 0
            test_epoch_count = 0

            for seqs, labels in val_dataloader:
                predictions = model(seqs.view(seqs.shape[0],-1))

                loss = criterion(predictions, labels)
                test_loss = criterion(predictions, labels)

                correct = predictions.argmax(axis=1) == labels
                acc = correct.sum().item() / correct.size(0)

                test_epoch_correct += correct.sum().item()
                test_epoch_count += correct.size(0)
                test_epoch_loss += loss.item()

    print(f"{epoch_loss=}")
    print(f"epoch accuracy: {epoch_correct / epoch_count}")
    print(f"{test_epoch_loss=}")
    print(f"valid epoch accuracy: {test_epoch_correct / test_epoch_count}")