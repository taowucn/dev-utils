import os, sys, argparse
import onnx
import onnxruntime
import torch
from torch import nn
#import torchvision
#from torchvision import datasets
#from torchvision import transforms

import numpy as np

device = "cpu"
print(f"Using {device} device")

#pth_file = "pth/ops_softmax.pth"
onnx_model_file = "onnx/ops_muladdi.onnx"

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.run_ops = nn.Sequential(
        )

    def forward(self, x):
        o = torch.mul(x, 2)
        output = torch.add(o, 1)
        return output


def export_to_pt2():
    model = NeuralNetwork().to(device)
    print(model)

    model.eval()
    input_names = ["input0"]
    output_names = ["output"]
    x = torch.arange(16).reshape(1, 1, 1, 16)

    outs = model(x)
    #print("input0: ", x)
    #print("input1: ", y)
    print("Pytorch out:", outs)

    out_model = "out_pt/op_muladdi.pt2"
    ep = torch.export.export(model, (x,))
    torch.export.save(ep, out_model)
    print(f"Export pt2 : {out_model}")


def export_to_onnx():
    model = NeuralNetwork().to(device)
    print(model)

    model.eval()
    input_names = ["input0"]
    output_names = ["output"]
    x = torch.arange(16).reshape(1, 1, 1, 16)

    outs = model(x)
    #print("input0: ", x)
    #print("input1: ", y)
    print("Pytorch out:", outs)
    onnx_model = onnx_model_file

    torch.onnx.export(model, x, onnx_model, export_params=True,
                      verbose = False,
                      input_names=input_names,
                      output_names=output_names)
    print(f"Export ONNX: {onnx_model}")


def infer_onnx():
    onnx_model = onnx.load(onnx_model_file)
    sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())

    x = torch.arange(16).reshape(1, 1, 1, 16).numpy()
    input_fd = {'input0' : x}
    outs = sess.run(None, input_fd)
    print("ONNX out:", outs)

    for arr, t in zip(outs, sess.get_outputs()):
        out_fn = "out_onnx_{}.bin".format(t.name)
        #print("output name: ", t.name, ", data:", arr)
        #arr.tofile(out_fn)
        #print("  save to bin file:", out_fn)


export_to_pt2()
export_to_onnx()
infer_onnx()
