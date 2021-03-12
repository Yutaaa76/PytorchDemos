import torch

# 自动扩张，without copying data
# Insert 1 dim ahead
# Expand dims with size 1 to same size

# Feature maps: [4,32,14,14]
# Bias: [32,1,1] => [1,32,1,1] => [4,32,14,14]
# 可用： 1. dim=1，扩张成相同的  2. no dim there， insert one dim and expand to same
# match from last dim

a = torch.rand(4, 32, 8)
b = torch.rand(1)
print(b.unsqueeze(0).unsqueeze(0).expand_as(a).shape)


