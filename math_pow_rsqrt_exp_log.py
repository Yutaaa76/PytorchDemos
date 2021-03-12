import torch

a = torch.full((2, 2), 3.)
print(a.pow(2))

print(a**2)

aa = a**2
print(aa.rsqrt())

print(aa**0.5)  # **0.25

# -----------------------------
a = torch.exp(torch.ones(2, 2))
print(a)
print(torch.log(a))


