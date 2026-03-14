import torch
import numpy as np

data = [[1,2], [3,4]]
#print(data)
x_data = torch.tensor(data)
#print(x_data)
#np_data = np.array(data)
#print(np_data)
#x_np = torch.from_numpy(np_data)
#print(x_np)
#x_ones = torch.ones_like(x_data)
#print(x_ones)
#
#x_rand = torch.rand_like(x_data, dtype=torch.bfloat16)
#print(x_rand)

x_ones = torch.ones_like(x_data)
print(x_ones)
x_ones = torch.ones(x_data.shape, dtype=int)
print(x_ones)

shape = (2,3)
rand_tensor = torch.rand(shape)
print(rand_tensor)
ones_tensor = torch.ones(shape)
print(ones_tensor)
zero_tensor = torch.zeros(shape)
print(zero_tensor)

print(ones_tensor.shape)
print(ones_tensor.dtype)
print(ones_tensor.device)

#zeros_tn
#print(rand_tensor)

device = 'cpu'
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()

ones_tensor = ones_tensor.to(device)
print(device)
print(ones_tensor.device)

    
