import torch

a = torch.rand(4, 1, 28, 28)
print("a:", a.shape)

# 参数是索引，正反索引
# 若tensor为n维， 参数范围为 -n ~ n-1
b = a.unsqueeze(0)  # 维度增加
print("b:", b.shape)

b = a.unsqueeze(-1)
print("b:", b.shape)

b = a.unsqueeze(4)
print("b:", b.shape)

b = a.unsqueeze(-4)
print("b:", b.shape)

# b = a.unsqueeze(5)
# print("b:", b.shape)

c = torch.tensor([1.2, 2.3])
print("c:", c)
print("c.shape:", c.shape)

d = c.unsqueeze(-1)
print(d)

d = c.unsqueeze(0)
print(d)

e = torch.rand(32)
f = torch.rand(4, 32, 14, 14)
e = e.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # 将f叠加在e上面，需要如此操作
print("e.shape: ", e.shape)

# squeeze 维度删减
print(e.squeeze().shape)  # 如果没有参数，则默认去掉维数为1的维度
print(e.squeeze(0).shape)
print(e.squeeze(-1).shape)
print(e.squeeze(1).shape)  # 32，维数不是1，所以不可以被squeeze()掉
print(e.squeeze(-4).shape)

