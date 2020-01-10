import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

n_pts=100
centers=[[-0.5,0.5],[0.5,-0.5]]
X,y=datasets.make_blobs(n_samples=n_pts,random_state=123,centers=centers,cluster_std=0.4)

x_data=torch.Tensor(X)
y_data=torch.Tensor(y.reshape(100,1))

class Model(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.linear=nn.Linear(input_size,output_size)
    def forward(self,x):
        return torch.sigmoid(self.linear(x))
torch.manual_seed(1)
model=Model(2,1)

[w,b]=model.parameters()
def get_params():
    w1,w2=w.view(2)
    return(w1.item(),w2.item(),b[0].item())

def plot_fit(title):
    plt.title=title
    plt.scatter(X[y==0,0],X[y==0,1])
    plt.scatter(X[y==1,0],X[y==1,1])
    w1,w2,b1=get_params()
    x1=np.array([-2,2])
    x2=(w1*x1+b1)/-w2
    plt.plot(x1,x2)
    plt.show(block=True)


#training part
Criterion=nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

epochs=1000
losses=[]
for i in range(1000):
    y_pred=model.forward(x_data)
    loss=Criterion(y_pred,y_data)
    print("epochs",i,"loss",loss.item())
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
plt.plot(range(epochs),losses)
plt.xlabel('loss')
plt.ylabel('epoch')
plt.show(block=True)



plot_fit("finalplot")
