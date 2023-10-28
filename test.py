import torch
import torch.nn as nn
import torch.optim as optim
from OfftargetDataset_train_balance import OfftargetDataset_train_balance 
from OfftargetDataset_val import OfftargetDataset_val
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import torch.nn.functional as F
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2,4, figsize=(10, 10))
fig.suptitle('on/off target')
autopct = "%.2f"

for i in list([0,1]):
    for j in list([1,2,3,4]):
        temp = np.loadtxt(f"./data/dataset{j+i*4}.csv", delimiter=',', skiprows=1, dtype=np.unicode_)
        temp = temp[:,0:3]
        result = temp[:,2:3]
        onTarget= result[result == "1"].size
        offTarget= result[result == "0"].size
        data=np.array([onTarget,offTarget])
        label = ["on","off"]
        axs[i,j-1].set_title("dataset{0}".format(j+i*4))
        axs[i,j-1].pie(data, labels=label,radius=1,autopct=autopct)  



fig.tight_layout()
plt.show()
        



