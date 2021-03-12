import torch

a = torch.rand(4, 32, 8)  # class1-4
b = torch.rand(5, 32, 8)  # class5-9
# 除了需要cat的维度，其余dim必须相等
print(torch.cat([a, b], dim=0).shape)

a1 = torch.rand(4, 3, 32, 32)
a2 = torch.rand(5, 3, 32, 32)
print(torch.cat([a1, a2], dim=0).shape)

a2 = torch.rand(4, 1, 32, 32)
print(torch.cat([a1, a2], dim=1).shape)

a1 = torch.rand(4, 3, 16, 32)
a2 = torch.rand(4, 3, 16, 32)
print(torch.cat([a1, a2], dim=2).shape)

# stack() 创建新的维度
print(torch.stack([a1, a2], dim=2).shape)
# torch.Size([4, 3, 2, 16, 32]) 第三个维度取0时，第四个维度为a1的数据，取2时，第四个维度为a2的数据

c = torch.rand(32, 8)  # 一个班，32人，8门课
d = torch.rand(32, 8)
print(torch.stack([c, d], dim=0).shape)
# torch.Size([2, 32, 8]): 两个班

f = torch.rand([30, 8])
# print(torch.stack([c, f], dim=0))
print(torch.cat([c, f], dim=0).shape)
