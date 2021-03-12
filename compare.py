import torch

a = torch.rand(4, 10) * 10 - torch.ones(4, 10) * 5

print(a > 0)
print(torch.gt(a, 0))  # greater = >
print(a != 0)

a = torch.ones(2, 3)
b = torch.rand(2, 3)
print(torch.eq(a, b))  # eq()比较每个元素是否相等
print(torch.eq(a, a))
print(torch.equal(a, a))  # equal()比较两个tensor是否相等
