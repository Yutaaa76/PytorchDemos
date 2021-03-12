import torch

a = torch.rand(3, 4)
b = torch.rand(4)
print(a+b)
print(torch.add(a, b))

print(torch.all(torch.eq(a-b, torch.sub(a, b))))
print(torch.all(torch.eq(a*b, torch.mul(a, b))))
print(torch.all(torch.eq(a/b, torch.div(a, b))))