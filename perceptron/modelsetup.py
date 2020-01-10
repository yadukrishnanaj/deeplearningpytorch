import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

n_pts=100
centers=[[-0.5,0.5],[0.5,-0.5]]
X,y=datasets.make_blobs(n_samples=n_pts,random_state=123,centers=centers,cluster_std=0.4)
def scatter_plot():
    plt.scatter(X[y==0,0],X[y==0,1])
    plt.scatter(X[y==1,0],X[y==1,1])
    plt.show(block=True)

class Model(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.linear=nn.Linear(input_size,output_size)
    def forward(self,x):
        return torch.sigmoid(self.linear(x))
torch.manual_seed(2)

model=Model(2,1)
[w,b]=model.parameters()
w1,w2=w.view(2)
def get_params():
    return (w1.item(),w2.item(),b[0].item())

def plot_fit(title):
    plt.title=title
    w1,w2,b1=get_params()
    x1=np.array([-2.0,2.0])
    x2=(w1*x1+b1)/-w2
    plt.plot(x1,x2,"r")
    scatter_plot()
    
   
    
    

plot_fit("ji")

