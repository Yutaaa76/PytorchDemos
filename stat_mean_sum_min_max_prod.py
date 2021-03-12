import torch

a = torch.arange(8).view(2, 4).float()
print(a)

print(a.min())
print(a.max())
print(a.mean())
print(a.prod())  # 累乘
print(a.sum())

print(a.argmax())  # 最大值所在索引
print(a.argmin())  # 最小值所在所以
# 注意这里输出的索引，是0和7，说明这类函数，不加参数的话，在运算时会把tensor打平，dim=1
a = a.view(1, 2, 4)
print(a)
print(a.argmin())
print(a.argmax())
a = torch.rand(2, 3, 4)
print(a.argmax())

print("---------------------------------------")
# 加参数:
a = torch.rand(4, 10)
print(a[0])
print(a.argmax())
print(a.argmax(dim=1))
print(a.argmax(dim=0))

