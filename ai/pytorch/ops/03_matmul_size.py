import torch

tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(10, 4, 5)
out = torch.matmul(tensor1, tensor2)
print("out shape:", out.size())


input = torch.randn(10, 3, 4)
mat2 = torch.randn(10, 4, 5)
res = torch.bmm(input, mat2)
print("out shape:", res.size())


tensor1 = torch.randn(1, 24, 2, 2)
tensor2 = torch.randn(6, 24, 2, 1)
out = torch.matmul(tensor1, tensor2)
print("out shape:", out.size())


tensor1 = torch.randn(6, 24, 1, 2)
tensor2 = torch.randn(1, 24, 2, 2)
out = torch.matmul(tensor1, tensor2)
print("out shape:", out.size())

