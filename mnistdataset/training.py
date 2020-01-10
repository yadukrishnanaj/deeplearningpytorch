import torch
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.nn.functional as F

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_dataset=datasets.MNIST(root='./data',train=True,download=True,transform=transform)

print(train_dataset)

training_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=100,shuffle=True)

def image_convert(tensor):
    image=tensor.clone().detach().numpy()
    image=image.transpose(1,2,0)
    image=image*np.array([0.5,0.5,0.5])/np.array([0.5,0.5,0.5])
    image=image.clip(0,1)
    return image



dataiter=iter(training_loader)
images,labels=dataiter.next()
fig=plt.figure(figsize=(25,4))


'''for idx in np.arange(20):
    ax=fig.add_subplot(2,10,idx+1)
    plt.imshow(image_convert(images[idx]))
    ax.set_title([labels[idx].item()])
    plt.show(block=True)'''

class Classifier(nn.Module):
    def __init__(self,din,h1,h2,dout):
        super().__init__()
        self.linear1=nn.Linear(din,h1)
        self.linear2=nn.Linear(h1,h2)
        self.linear3=nn.Linear(h2,dout)
        
    def forward(self,x):
        x=F.relu(self.linear1(x))
        x=F.relu(self.linear2(x))
        x=self.linear3(x)
        return x

model=Classifier(784,125,65,10)
print(model)


Criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

running_correct_history=[]
epochs=12
running_loss_history=[]
for i in range(epochs):
    running_loss=0.0
    running_correct=0.0
    for inputs,labels in training_loader:
        inputs=inputs.view(inputs.shape[0],-1)
        outputs=model(inputs)
        loss=Criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    else:
        epochs_loss=running_loss/len(training_loader)
        running_loss_history.append(epochs_loss)
        print("training loss {:.4f}".format(epochs_loss))


  

        
