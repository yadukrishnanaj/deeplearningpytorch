import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
x=torch.randn(100,1)*10
y=x+torch.randn(100,1)*3

class LR(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.linear=nn.Linear(input_size,output_size)
    def forward(self,x):
        pred=self.linear(x)
        return pred
torch.manual_seed(1)
model=LR(1,1)
[w,b]=model.parameters()

def get_params():
    return (w[0][0].item(),b[0].item())

def plot_fit(title):
    plt.title=title
    w1,b1=get_params()
    x1=np.array([-30,30])
    y1=w1*x1+b1
    plt.plot(x.numpy(),y.numpy(),'o')
    plt.plot(x1,y1)
    plt.show(block=True)

Criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

losses=[]
epochs=100
 
for i in range(epochs):
    y_predict=model.forward(x)
    loss=Criterion(y_predict,y)
    print("epochs:",i,"loss:",loss.item())
    losses.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plot_fit("correct")
#plt.plot(range(epochs),losses)
#plt.ylabel('yloss')
#plt.xlabel('epochs')
#plt.show(block=True)
