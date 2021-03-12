import torch
import numpy as np

a = torch.rand(4, 3, 28, 28)
print("a = ", a)
print(a[0].shape)
print(a[0, 0].shape)
print(a[0, 0, 2, 2])

# 取连续的片段
print(a[:2].shape)
print(a[:2, :1, :, :].shape)  # :1  从第头到第1
print(a[:2, 1:, :, :].shape)  # 1:  从第1到最末尾
print(a[:2, -1:, :, :].shape)  # -1: 只取了最末尾   反向索引，比如正向为 0,1,2  反向为 -3，-2，-1

# :单独出现，表示 取全部
# :带一个数字， ：n 不包含第n个   n: 从第n个到最末尾
# start:end  [start,end)这个区间
# 0:28:3 两个::同时出现， 0:28 == 0:28:1
# ::2 == 0:28:2
# ::1 == :

# 按一定间隔选取
print(a[:, :, 0:28:2, 0:28:2].shape)


# 给具体的索引
print(a.index_select(0, torch.tensor([0, 2])).shape)
print(a.index_select(1, torch.tensor([0, 2])).shape)

b = torch.linspace(1, 12, steps=12).view(3, 4)
print("b = ", b)
print(torch.index_select(b, 0, torch.tensor([0, 2])))
print(torch.index_select(b, 1, torch.tensor([0, 2])))
# 第一个参数 b 为索引的对象，
# 第二个0表示按行索引，1表示按列
# 第三个表示索引的序号，用list格式，且必须用一个tensor来传输

# a = torch.rand(4, 3, 28, 28)
print(b.index_select(1, torch.arange(4)))

# ...
print(a[...].shape)
print(a[0, ...].shape)
print(a[:, 1, ...])
print(a[:, 1, ...].shape)
print(a[..., :2].shape)
