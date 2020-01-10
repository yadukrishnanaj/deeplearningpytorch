import torch
from torch.nn import Linear
torch.manual_seed(1)
model=Linear(in_features=1,out_features=1)
print(model.bias,model.weight)
x=torch.tensor([2.0])
print(model(x))
x=torch.tensor([[2.0],[3.0]])
print(model(x))
