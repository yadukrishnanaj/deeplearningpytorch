import torch
print(torch.cuda.is_available())

print(torch.cuda.current_device)

print(torch.cuda.get_device_name(0))
#print which GPU

print(torch.cuda.memory_allocated())

print(torch.cuda.memory_cached())


a=torch.FloatTensor([1.0,2.0])

print(a.device)

a=torch.FloatTensor([1.0,2.0]).cuda()

print(a.device)
 

print(torch.cuda.memory_allocated())

