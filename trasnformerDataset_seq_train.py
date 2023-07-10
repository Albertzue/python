import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import torch
def Convert(result):
    return  1.0 if(result == "on-target") else 0.0
 
def toNum(seq):

    dict = {"A": 0, "G": 1, "C": 2, "T":3}
    result = []
    for i in  enumerate(seq):
        temp = dict[i[1]]
        result.append(temp)
    
    return result

class OfftargetDataset_train(Dataset):
    def __init__(self):
        x = []
        y = []
       
        for i in list([1]):
            temp = np.loadtxt(f"./data/dataset{i+1}.csv", delimiter=',', skiprows=1, dtype=np.unicode_)
            temp = temp[:,0:3]
              ##SOS 0 EOS -1
            for row_num in range(temp.shape[0]):
                otdna_seq = toNum(temp[row_num][1])
                sgrna_seq = toNum(temp[row_num][0])
                temp_x = np.concatenate((otdna_seq,sgrna_seq),axis=0) 
                y.append(float(temp[row_num][2]))
                x.append(temp_x)    

            print(f"data{i+1} loaded")
            
        x = np.asarray(x,dtype=np.int64)
        y = np.asarray(y,dtype=np.int64)
        self.x =torch.from_numpy(x)
        self.y =torch.from_numpy(y)
        self.n_samples = len(x)

        

    def __len__(self):
       return self.n_samples

    def __getitem__(self, index):
       return self.x[index], self.y[index]