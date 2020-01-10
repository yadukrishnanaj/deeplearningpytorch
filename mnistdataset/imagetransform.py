import torch
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np

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


for idx in np.arange(20):
    ax=fig.add_subplot(2,10,idx+1)
    plt.imshow(image_convert(images[idx]))
    ax.set_title([labels[idx].item()])
    plt.show(block=True)

