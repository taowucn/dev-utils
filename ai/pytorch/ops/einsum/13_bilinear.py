import torch
import numpy as np

a = torch.randn(2,3)
b = torch.randn(5,3,7)
c = torch.randn(2,7)
# i = 2, k = 3, j = 5, l = 7
torch_ein_out = torch.einsum('ik,jkl,il->ij', [a, b, c]).numpy()
m = torch.nn.Bilinear(3, 7, 5, bias=False)
m.weight.data = b
torch_org_out = m(a, c).detach().numpy()

print("input a:\n", a)
print("input b:\n", b)
print("torch ein out: \n", torch_ein_out)
print("torch org out: \n", torch_org_out)
print("is torch_org_out == torch_ein_out ?", np.allclose(torch_ein_out, torch_org_out))