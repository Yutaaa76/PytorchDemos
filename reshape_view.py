import torch

a = torch.rand(4, 1, 8, 8)
print(a)
print(a.shape)

print(a.view(4, 8 * 8))
print(a.view(4, 8 * 8).shape)

print(a.view(4 * 8, 8))
print(a.view(4 * 8, 8).shape)

print(a.view(4 * 1, 8, 8))
print(a.view(4 * 1, 8, 8).shape)

print(a.view(4, 64))
print(a.view(4, 64).shape)

print(a.view(4, 8, 8, 1))
print(a.view(4, 8, 8, 1).shape)


