import numpy as np
import torch

a = np.array([2., 3.3])
print(a)
b = torch.from_numpy(a)
print(b)

c = np.ones([2, 3])
print(c)
d = torch.from_numpy(c)
print(d)

e = torch.tensor([2., 3.2])  #使用list作为参数
print(e)

f = torch.FloatTensor([2., 3.2])
print(f)

g = torch.tensor([[2., 3.2], [1., 22.3]])
print(g)

h = torch.Tensor(2)  #Tensor和tensor不同， Tensor和FloatTensor用法类似,一般直接接收数据的维度，也可接受数据，但必须用list表示
print(h)

i = torch.empty(2)  #分配空的内存空间，参数为shape

j = torch.IntTensor(3, 4, 4)
print(j)

k = torch.tensor([1.2, 3])
# set default tensor type: torch.set_default_tensor_type(torch.xxxTensor)指定Tensor的类型
print(k, k.type())
torch.set_default_tensor_type(torch.DoubleTensor)
k = torch.tensor([1, 3])
print(k, k.type())

# 随机初始化 rand/rand_like, randint, randn
l = torch.rand(3, 3)  #在(0,1)区间中随机均匀分布
print(l)

m = torch.rand_like(l)  #参数为tensor，将其shape读出来后，送给rand()
print(m)

n = torch.randint(1, 20, [9, 9])  #在[1, 19]区间中随机均匀分布,取不到 20
print(n)

o = torch.randn(9, 9)  #正态分布 N(0, 1)
print("o = ", o)
# 还有torch.normal()

p = torch.full([ ], 3.1415926)
print("p = ", p)

q = torch.full([2, 3], 3.14159)
print("q = ", q)

# 生成等差数列 arange/range (阶梯式)
r = torch.arange(0, 10)
print("r = ", r)
r = torch.arange(0, 10, 2)
print("r = ", r)

s = torch.range(0, 10)  #不建议使用
print("s = ", s)

# linspace/logspace
# linspace 等差数列
t = torch.linspace(0, 10, 10)  #分成10个数
print("t = ", t)
t = torch.linspace(0, 10, 11)  #分成11个数,正好等分
print("t = ", t)
# logspace 等比数列
u = torch.logspace(0, -1, steps=10)  #从10的0次方为起始值，10的-1次方为终止值，10个数的等比数列
print("u = ", u)
u = torch.logspace(12, -1, steps=10)
print("u = ", u)

# ones/zeros/eye
v = torch.ones(4, 4)
print("v = ", v)
v = torch.zeros(4,4)
print("v = ", v)
v = torch.ones_like(v)
print("v = ", v)
v = torch.eye(4, 4)  #左上角到右下角
print("v = ", v)
v = torch.eye(3, 4)
print("v = ", v)
# eye()只能有1个或2个参数，仅适用于矩阵
v = torch.eye(3) # 等于输入（3,3）
print("v = ", v)

# randperm()
print(torch.randperm(10))  #生成随机索引,[0,10)区间
x = torch.rand(2, 3)
y = torch.rand(2, 2)
print(x, y)
idx = torch.randperm(2)
print(idx)
print(x[idx])
print(y[idx])


