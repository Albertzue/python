import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import torch
def Convert(result):
    return  1.0 if(result == "on-target") else 0.0
 
def toOneHot(seq):
    dict = {"A": np.array([1,0,0,0]).T, "G": np.array([0,1,0,0]).T, "C": np.array([0,0,1,0]).T, "T":np.array([0,0,0,1]).T}
    result = []
    for i in  enumerate(seq):
        temp = dict[i[1]]
        result.append(temp)
    
    return result

class OfftargetDataset_train(Dataset):
    def __init__(self,convertType='onehot'):
        x = []
        y = []
       
        for i in list([0,]):
            temp = np.loadtxt(f"./data/dataset{i+1}.csv", delimiter=',', skiprows=1, dtype=np.unicode_)
            temp = temp[:,0:3]

            for row_num in range(temp.shape[0]):
                if convertType == 'onehot':
                    otdna_onehot = toOneHot(temp[row_num][1])
                    sgrna_onehot = toOneHot(temp[row_num][0])
                    temp_x = np.concatenate((otdna_onehot,sgrna_onehot),axis=1) 
                    y.append(int(temp[row_num][2]))
                    x.append(temp_x.T)    

            print(f"data{i+1} loaded")
            
        x = np.asarray(x,dtype=np.float32)
        y = np.asarray(y,dtype=np.float32)
        self.x =torch.from_numpy(x)
        self.y =torch.from_numpy(y)
        self.n_samples = len(x)

        

    def __len__(self):
       return self.n_samples

    def __getitem__(self, index):
       return self.x[index], self.y[index]