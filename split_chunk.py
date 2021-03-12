import torch

# splic(): by lenth
a = torch.rand(32, 8)
b = torch.rand(32, 8)

c = torch.stack([a, b, b], dim=0)
print(c.shape)

aa, bb = c.split([2, 1], dim=0)  # 0,1,2三个班级，0和1在一起，2单独
print(aa.shape, bb.shape)

aa, bb, cc = c.split([1, 1, 1], dim=0)
print(aa.shape, bb.shape, cc.shape)

aa, bb = c.split(2, dim=0)  # 等长拆分，最后一份如果<=0，会报错
print(aa.shape, bb.shape)

aa, bb, cc = c.split(1, dim=0)
print(aa.shape, bb.shape, cc.shape)

# chunk(): by num
aa, bb = c.chunk(2, dim=0)
print(aa.shape, bb.shape)