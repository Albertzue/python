import torch
import torch.nn as nn
import torch.optim as optim
from OfftargetDataset_train import OfftargetDataset_train
from OfftargetDataset_val import OfftargetDataset_val
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import torch.nn.functional as F
learning_rate = 1e-2
dropout = 0.2
loss_fn = nn.CrossEntropyLoss()
batch_size = 128
n_epochs = 50
train_set = OfftargetDataset_train()
val_set = OfftargetDataset_val()
# generator = torch.Generator().manual_seed(42)
# dataset = OfftargetDataset_train()
# train_set, val_set = torch.utils.data.random_split(dataset,[0.8,0.2], generator= generator)
generator = torch.Generator().manual_seed(66)
train_dataloader = DataLoader(batch_size=batch_size, shuffle=True, dataset=train_set)
val_dataloader = DataLoader(batch_size=batch_size, shuffle=False, dataset=val_set)
path ="nnModel/cnn"



class CNN_model(nn.Module):
   def __init__(self):
       super().__init__()
       self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
       self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
       self.fc1 = nn.Linear(640, 64)
       self.fc2 = nn.Linear(64, 2)
       self.dropout =nn.Dropout()

   def forward(self, x):
       out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
       out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
       out = out.view(-1, 64*5*2)
       out = torch.relu(self.fc1(out))
       out = self.dropout(out)
       out = self.fc2(out)
       return out

model = CNN_model()

optimizer = optim.Adam(model.parameters(),lr = learning_rate)

for epoch in range(n_epochs):
    for seqs, labels in train_dataloader:
        outputs = model(seqs.view(seqs.shape[0],1,8,-1))

        loss =loss_fn(outputs,labels.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if(epoch % 10 ==0):
        print(f"Epoch{epoch} loss={loss.item()}")
        
correct = 0
scores = np.array([])
total =0
with torch.no_grad():
    for seqs, labels in val_dataloader:
        outputs = model(seqs.view(seqs.shape[0],1,8,-1))    
        data,predicted = torch.max(outputs,dim = 1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
        scores = np.concatenate((scores, data.numpy()))
print(f"Accuracy:{correct/total}")

torch.save(model,path)