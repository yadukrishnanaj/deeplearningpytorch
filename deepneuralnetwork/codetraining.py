import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

n_pts=500
centers=[[-0.5,0.5],[0.5,-0.5]]
X,y=datasets.make_circles(n_samples=n_pts,random_state=123,noise=0.1,factor=0.2)

x_data=torch.Tensor(X)
y_data=torch.Tensor(y.reshape(500,1))
def scatter_plot():
    plt.scatter(X[y==0,0],X[y==0,1])
    plt.scatter(X[y==1,0],X[y==1,1])
    plt.show(block=True)

class Model(nn.Module):
    def __init__(self,input_size,H1,output_size):
        super().__init__()
        self.linear1=nn.Linear(input_size,H1)
        self.linear2=nn.Linear(H1,output_size)
    def forward(self,x):
        x=torch.sigmoid(self.linear1(x))
        x=torch.sigmoid(self.linear2(x))
        return x

torch.manual_seed(2)
model=Model(2,4,1)

Criterion=nn.BCELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
epochs=1000
losses=[]
for i in range(epochs):
    y_pred=model.forward(x_data)
    loss=Criterion(y_pred,y_data)
    print("epochs",i,"loss",loss)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#plt.plot(range(epochs),losses)
#plt.show(block=True)

def plot_decision_boundary(X,y):
    X_span=np.linspace(min(X[:,0]-0.25),max(X[:,0]+0.25))
    y_span=np.linspace(min(X[:,1]-0.25),max(X[:,1]+0.25))
    xx,yy=np.meshgrid(X_span,y_span)
    grid=torch.Tensor(np.c_[xx.ravel(),yy.ravel()])
    pred_func=model.forward(grid)
    Z=pred_func.view(xx.shape).detach().numpy()
    plt.contourf(xx,yy,Z)

plot_decision_boundary(X,y)
scatter_plot()




