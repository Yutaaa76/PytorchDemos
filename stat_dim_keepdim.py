import torch

a = torch.rand(4, 10)
print(a)

print(a.max(dim=1))
print(a.argmax(dim=1))

print(a.max(dim=1, keepdim=True))
print(a.argmax(dim=1, keepdim=True))