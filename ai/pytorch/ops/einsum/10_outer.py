import torch
import numpy as np

a = torch.arange(3)
b = torch.arange(3,7)  # [3, 4, 5, 6]
# i = 3, j = 4
torch_ein_out = torch.einsum('i,j->ij', [a, b]).numpy()
# 等价形式，可以省略箭头和输出
#torch_ein_out2 = torch.einsum('i,j', [a, b]).numpy()
torch_org_out = torch.outer(a, b).numpy()

print("input a:\n", a)
print("input b:\n", b)
print("torch ein out: \n", torch_ein_out)
print("torch org out: \n", torch_org_out)
print("is torch_org_out == torch_ein_out ?", np.allclose(torch_ein_out, torch_org_out))