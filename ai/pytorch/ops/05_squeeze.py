import torch

class Net(torch.nn.Module):
   def __init__(self):
      super().__init__()

   def forward(self, x):
      return torch.squeeze(x, dim=2)

net = Net()
torch.onnx.export(net, torch.ones(1,3,1,2), 'onnx/ops_squeeze.onnx', opset_version=11)
