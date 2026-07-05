import torch
import numpy as np

a = torch.randn(2,3,4,5,6)
# i = 5, j = 6
torch_ein_out = torch.einsum('...ij->...ji', [a]).numpy()
torch_org_out = a.permute(0, 1, 2, 4, 3).numpy()

#print("input:\n", a)
#print("torch ein out: \n", torch_ein_out)
#print("torch org out: \n", torch_org_out)
print("is torch_org_out == torch_ein_out ?", np.allclose(torch_ein_out, torch_org_out))