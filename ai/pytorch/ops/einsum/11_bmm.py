import torch
import numpy as np

a = torch.randn(2,3,5)
b = torch.randn(2,5,4)
# i = 2, j = 3, k = 5, l = 4
torch_ein_out = torch.einsum('ijk,ikl->ijl', [a, b]).numpy()
torch_org_out = torch.bmm(a, b).numpy()

print("input a:\n", a)
print("input b:\n", b)
print("torch ein out: \n", torch_ein_out)
print("torch org out: \n", torch_org_out)
print("is torch_org_out == torch_ein_out ?", np.allclose(torch_ein_out, torch_org_out))