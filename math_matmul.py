import torch

a = torch.tensor([[3., 3.], [3., 3.]])
b = torch.ones(2, 2)
print(a)
print(b)
print("----------------")
print(a*b)  #相同位置元素相乘
print(torch.mm(a, b))  # only for 2D, 不推荐
print(torch.matmul(a, b))  # 推荐
print(a@b)  # @ = matmul

z = torch.rand(4, 784)
x = torch.rand(4, 784)
w = torch.rand(512, 784)
print((x@w.t()).shape)
# x@w.t()用来降维
# t()只适用于2D，如果高维需要用transpose()

a = torch.rand(4, 3, 28, 64)
b = torch.rand(4, 3, 64, 32)
print(torch.matmul(a, b).shape)

b = torch.rand(4, 1, 64, 32)
# [4,3,28,64]
# [4,1,64,32]
# 64被消掉，1可以通过broadcast扩展为3
print(torch.matmul(a, b).shape)