import torch
import numpy as np

a = torch.arange(6).reshape(2, 3)
b = torch.arange(15).reshape(3, 5)
# i = 2, k = 3, j = 5
torch_ein_out = torch.einsum('ik,kj->ij', [a, b]).numpy()
# 等价形式，可以省略箭头和输出
#torch_ein_out2 = torch.einsum('ik,kj', [a, b]).numpy()
torch_org_out = torch.mm(a, b).numpy()

print("input a:\n", a)
print("input b:\n", b)
print("torch ein out: \n", torch_ein_out)
print("torch org out: \n", torch_org_out)
print("is torch_org_out == torch_ein_out ?", np.allclose(torch_ein_out, torch_org_out))