import torch

a = torch.tensor(3.14)

print(a.floor())  # 向下取整
print(a.ceil())  # 向上取整
print(a.trunc())  # 取整数
print(a.frac())  # 取小数

a = torch.tensor(3.499)
print(a.round())  # 四舍五入
a = torch.tensor(3.5)
print(a.round())

# clamp(): gradient clipping 梯度裁剪
grad = torch.rand(2, 3) * 15
print(grad.max())
print(grad.median())
print(grad.clamp(10))  # 最小为10
print(grad)
print(grad.clamp(0, 10))  # 最小0，最大10

