
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SVMSMOTE
import pandas as pd


def Convert(result):
    return  1.0 if(result == "on-target") else 0.0
 
def toOneHot(seq):
    dict = {"A": np.array([1,0,0,0]).T, "G": np.array([0,1,0,0]).T, "C": np.array([0,0,1,0]).T, "T":np.array([0,0,0,1]).T}
    result = np.asarray([])
    for i in  enumerate(seq):
        temp = dict[i[1]]# how to export 1d data total
        result = result.append(temp)
     
    result = np.asarray(result)  
    return result


data=[]
for i in list([1,2,3,4,5,6,7,8]):
        df = pd.read_csv(f"./data/dataset{i+1}.csv", delimiter=',', header=0 , dtype=np.unicode_)
        df["otdna"] = df['otdna'] + df['sgrna']
        temp = np.array(df)
        temp = temp[:,1:3]
          
        x = []
        y = []
        
        x_temp = temp[:,0:1]
        fun = np.frompyfunc(toOneHot, 1, 1)
        x = fun(x_temp)    
        
        
        # for row_num in range(temp.shape[0]):
        #     otdna_onehot = test[row_num][1]
        #     sgrna_onehot = test[row_num][0]
        #     temp_x = np.concatenate((otdna_onehot,sgrna_onehot),axis=1) 
        #     x.append(temp_x.T)
        #     y.append(int(temp[row_num][2]))
            
        sm = SVMSMOTE(random_state=42)
        rx= np.reshape(x,(len(x),184))
        X_res, y_res = sm.fit_resample(rx, y) 
        data.append([y_res[y_res == 1], y_res[y_res == 0]])




fig, axs = plt.subplots(2,4, figsize=(10, 10))
fig.suptitle('on/off target')
autopct = "%.2f"

for i in list([0,1]):
    for j in list([1,2,3,4]):
        temp = np.loadtxt(f"./data/dataset{j+i*4}.csv", delimiter=',', skiprows=1, dtype=np.unicode_)
        temp = temp[:,0:3]
        result = temp[:,2:3]
        label = ["on","off"]
        axs[i,j-1].set_title("dataset{0}".format(j+i*4))
        axs[i,j-1].pie(data[j+i*4 -1], labels=label,radius=1,autopct=autopct)  



fig.tight_layout()
plt.show()
        




   
   
   
   