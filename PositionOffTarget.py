import torch
import torch.nn as nn
import torch.optim as optim
from trasnformerDataset_seq_train import OfftargetDataset_train
from trasnformerDataset_seq_val import OfftargetDataset_val
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import torch.nn.functional as F
from positionModel import PositionNN 

n_epochs = 100
batch_size = 64
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PositionNN(num_tokens=4, dim_model=16).to(device)
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# generator = torch.Generator().manual_seed(66)
# dataset = OfftargetDataset_train()
# train_set, val_set = torch.utils.data.random_split(dataset,[0.8,0.2], generator= generator)
train_set = OfftargetDataset_train()
val_set = OfftargetDataset_val()

train_dataloader = DataLoader(batch_size=batch_size, shuffle=True, dataset=train_set)
val_dataloader = DataLoader(batch_size=batch_size, shuffle=False, dataset=val_set)
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

for epoch in range(n_epochs):
    for seqs, labels in train_dataloader:
        outputs = model(seqs.view(seqs.shape[0],-1))
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
        outputs = model(seqs.view(seqs.shape[0],-1))
        data,predicted = torch.max(outputs,dim = 1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
        scores = np.concatenate((scores, data.numpy()))
print(f"Accuracy:{correct/total}")

y = [label for seq, label in val_set]
y = np.array(y)
fpr,tpr, thresholds = metrics.roc_curve(y,scores)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = f'FNN AUC={roc_auc}')

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()