import torch

tensor = torch.tensor([[1,2],[3,4]])
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
#print(f"{y1=}")
#print(f"{y2=}")
y3 = torch.rand_like(y1, dtype=torch.float16)
#print(y3)
torch.mul(tensor, tensor, out=y3)
#print(y3)

agg = tensor.sum()
item = agg.item()
#print(f'{tensor=}')
#print(f'{agg=}')
#print(f'{item=}')

#tensor.add_(5)
#tensor += 5
#print(f'{tensor=}')
#print(tensor)

print(f"{tensor=}")
n = tensor.numpy()
print(f"{n=}")

tensor += 100

print(f"{tensor=}")
print(f"{n=}")
