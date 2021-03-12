import torch

# transpose(1, 3): 原来是[b, c, H, W],变成[b, W, H, c]
a = torch.rand(4, 3, 32, 32)
# a1 = a.transpose(1, 3).view(4, 3 * 32 * 32).view(4, 3, 32, 32)
print(a.transpose(1, 3).shape)
a1 = a.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 3, 32, 32)  # 错的，这样又把c提前了，会造成数据污染
print("a1.shape: ", a1.shape)
a2 = a.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 32, 32, 3).transpose(1, 3)
print("a2.shape: ", a2.shape)

print(torch.all(torch.eq(a, a1)))
print(torch.all(torch.eq(a, a2)))


# b = torch.rand(3, 4, 2)
# print(b)
# print(b.transpose(0, 2))
# print(b.transpose(0, 2).contiguous())

# permute(),参数直接写变换后相对于之前的顺序
print(a.transpose(1, 3).transpose(1, 2).shape)
print(a.permute(0, 2, 3, 1).shape)

