import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data
from offtargetDataset import OfftargetDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 2


dataset = OfftargetDataset()
generator = torch.Generator().manual_seed(42)
train_set, val_set = torch.utils.data.random_split(dataset,[0.8,0.2], generator= generator)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size= batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=val_set,
                                           batch_size= batch_size,
                                           shuffle=False)

input_size =  4
hidden_size = 200
num_classes = 2
num_epochs = 5

learning_rate = 0.001

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.sigmoid(out)
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes= num_classes)

criterion = nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)



n_total_steps = len(train_loader)

#y_train = y_train.view(y_train.shape[0],1)
#y_test = y_test.view(y_test.shape[0],1)


for epoch in range(num_epochs):
    for i,(seqs, labels) in enumerate(train_loader):
        outputs = model(seqs)
        loss = criterion(outputs, labels)
        
       
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f'epoch:{epoch+1}/{num_epochs} , step:{i}/{n_total_steps-1} loss:{loss.item()}')
        

with torch.no_grad():
    for i, (seqs,labels) in enumerate(test_loader):
        outputs = model(seqs)
        
        print(f"index:{i} pred:{outputs} answer:{labels[1]}")