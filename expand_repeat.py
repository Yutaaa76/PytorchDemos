import torch

# b [1, 32, 1, 1] 经过扩展变成 [4, 32, 14, 14]
a = torch.rand(4, 32, 14, 14)
b = torch.rand(1, 32, 1, 1)
print(b.expand(4, 32, 14, 14).shape)
print(b.expand(-1, 32, -1, -1).shape)
print(b.expand(-1, 32, -1, -4).shape)  # 非-1的负数可以写上去，但是没有意义，是个bug

# repeat()的参数不代表需要扩张的维度，而是每个维度要拷贝的倍数
print(b.repeat(4, 32, 1 , 1).shape)
print(b.repeat(4, 1, 1, 1).shape)
print(b.repeat(4, 1, 32, 32).shape)
