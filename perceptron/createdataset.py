import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import torch
n_pts=100
centers=[[-0.5,0.5],[0.5,-0.5]]
X,y=datasets.make_blobs(n_samples=n_pts,random_state=123,centers=centers,cluster_std=0.4)#
#print(X)

plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.show(block=True)

x_data=torch.tensor(X)
y_data=torch.tensor(y)

