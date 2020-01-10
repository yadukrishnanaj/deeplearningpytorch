import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt



#model intialisation
class Model(nn.Module):
    def __init__(self,in_features=4,h1=8,h2=9,out_features=3):
        super().__init__()
        self.fc1=nn.Linear(in_features,h1)
        self.fc2=nn.Linear(h1,h2) 
        self.out=nn.Linear(h2,out_features)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.out(x)
        return x
torch.manual_seed(32)

model=Model()
#print(next(model.parameters()).is_cuda)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_model=model.to(device)

#print(next(gpu_model.parameters()).is_cuda)

df=pd.read_csv('iris.csv')
X=df.drop('target',axis=1).values
y=df['target'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=33)

X_train = torch.FloatTensor(X_train).cuda()
X_test = torch.FloatTensor(X_test).cuda()
y_train = torch.LongTensor(y_train).cuda()
y_test = torch.LongTensor(y_test).cuda()


trainloader=DataLoader(X_train,batch_size=60,shuffle=True,pin_memory=True)
testloader=DataLoader(X_test,batch_size=60,shuffle=False,pin_memory=True)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

import time
start=time.time()
epochs=100
losses=[]


for i in range(epochs):
    y_pred=gpu_model.forward(X_train)
    loss=criterion(y_pred,y_train)
    losses.append(loss)
    print("epochs:",i,"loss:",loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("totaltimefor training",time.time()-start)

plt.plot(range(epochs),losses)
plt.show(block=True)   



     


