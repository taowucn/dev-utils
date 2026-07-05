import torch
import numpy as np

a = torch.arange(6).reshape(2, 3)
# i = 2, j = 3
torch_ein_out = torch.einsum('ij->j', [a]).numpy()
torch_org_out = torch.sum(a, dim=0).numpy()

print("input:\n", a)
print("torch ein out: \n", torch_ein_out)
print("torch org out: \n", torch_org_out)
print("is torch_org_out == torch_ein_out ?", np.allclose(torch_ein_out, torch_org_out))
