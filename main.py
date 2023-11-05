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


fnnpath ="nnModel/fnn_lof"
cnnpath ="nnModel/cnn"
learning_rate = 1e-2
dropout = 0.1
loss_fn = nn.CrossEntropyLoss()
batch_size = 64
n_epochs = 100
generator = torch.Generator().manual_seed(42)
dataset = OfftargetDataset_train()
train_set, val_set = torch.utils.data.random_split(dataset,[0.8,0.2], generator= generator)
# train_set = OfftargetDataset_train()
# val_set = OfftargetDataset_val()
train_dataloader = DataLoader(batch_size=batch_size, shuffle=True, dataset=train_set)
val_dataloader = DataLoader(batch_size=batch_size, shuffle=False, dataset=val_set)



fnnModel = torch.load(fnnpath)
fnnModel.eval()
fnncorrect = 0
fnnscores = np.array([])
fnntotal =0
with torch.no_grad():
    for seqs, labels in val_dataloader:
            outputs = fnnModel(seqs.view(seqs.shape[0],-1))    
            data,predicted = torch.max(outputs,dim = 1)
            fnntotal += labels.shape[0]
            fnncorrect += int((predicted == labels).sum())
            fnnscores = np.concatenate((fnnscores, data.numpy()))


cnnModel = torch.load(cnnpath)
cnnModel.eval()
cnncorrect = 0
cnnscores = np.array([])
cnntotal =0
with torch.no_grad():
    for seqs, labels in val_dataloader:
        outputs = cnnModel(seqs.view(seqs.shape[0],1,8,-1))    
        data,predicted = torch.max(outputs,dim = 1)
        cnntotal += labels.shape[0]
        cnncorrect += int((predicted == labels).sum())
        cnnscores = np.concatenate((cnnscores, data.numpy()))
        
print(f"FNN_Accuracy:{fnncorrect/fnntotal}")
print(f"CNN_Accuracy:{cnncorrect/cnntotal}")
  
   
   
   
        
y = [label for seq, label in val_set]
y = np.array(y)





fnn_fpr,fnn_tpr, fnn_thresholds = metrics.roc_curve(y, fnnscores)
fnn_roc_auc = metrics.auc(fnn_fpr, fnn_tpr)




plt.plot(fnn_fpr, fnn_tpr, color='b', label = f'FNN AUC={fnn_roc_auc}')


cnn_fpr, cnn_tpr, cnn_thresholds = metrics.roc_curve(y, cnnscores)
cnn_roc_auc = metrics.auc(cnn_fpr, cnn_tpr)
plt.plot(cnn_fpr, cnn_tpr, color='r', label = f'CNN AUC={cnn_roc_auc}')


plt.title('Receiver Operating Characteristic')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()