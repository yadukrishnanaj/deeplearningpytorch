import numpy as np
import torch
import matplotlib.pyplot as plt

x=torch.linspace(1,10)
y=torch.sin(x)
plt.plot(x.numpy(),y.numpy())
plt.show(block=True)

