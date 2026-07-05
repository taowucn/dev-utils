import torch
import numpy as np

a = torch.randn(2,3,5,7)
b = torch.randn(11,13,3,17,5)
# p = 2, q = 3, r = 5, s = 7
# t = 11, u = 13, v = 17, r = 5
torch_ein_out = torch.einsum('pqrs,tuqvr->pstuv', [a, b]).numpy()
torch_org_out = torch.tensordot(a, b, dims=([1, 2], [2, 4])).numpy()

print("input a:\n", a)
print("input b:\n", b)
print("torch ein out: \n", torch_ein_out)
print("torch org out: \n", torch_org_out)
print("is torch_org_out == torch_ein_out ?", np.allclose(torch_ein_out, torch_org_out))