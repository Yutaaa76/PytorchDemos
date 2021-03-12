import torch

a = torch.full([8], 1.)
b = a.view(2, 4)
c = a.view(2, 2, 2)
print(b)
print(c)

# norm-p
# 第一范数：所有元素绝地址的和
print(a.norm(1))
print(b.norm(1))
print(c.norm(1))

# 第二范数：所有元素平方和再开根号
print(a.norm(2))
print(b.norm(2))
print(c.norm(2))

print("---------------------------------")
# 指定维度的范数
print(b.norm(1, dim=1))
print(b.norm(2, dim=1))
print(c.norm(1, dim=1))
print(c.norm(1, dim=0))
print(c.norm(2, dim=0))

