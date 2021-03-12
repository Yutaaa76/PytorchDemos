import torch

a = torch.rand(4, 10)
print(a)

print(a.topk(5, dim=1))  # 递减
print(a.topk(5, dim=1, largest=False))  # 递增

# 递增,返回第k小的
print(a.kthvalue(10, dim=1))
print(a.kthvalue(3))
print(a.kthvalue(3, dim=1))
print(a.kthvalue(3, dim=1, keepdim=True))
