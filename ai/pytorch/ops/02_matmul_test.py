import torch

# batched matrix x broadcasted matrix
a = torch.randn(6, 24, 2, 1)
b = torch.randn(1, 24, 2, 2)

c = torch.matmul(b, a)
print(c.size())
