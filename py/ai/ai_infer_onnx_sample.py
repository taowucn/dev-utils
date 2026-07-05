#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import onnx
import onnxruntime as rt
from onnx import helper

onnx_model = onnx.load("sample.onnx")  ## TODO for sample.onnx
graph = onnx_model.graph

new_outs = ['823', '857', '883', '943', '994']  ## TODO for intermediate data
# append intermediate outputs:
#graph_outputs = [helper.make_empty_tensor_value_info(t) for t in new_outs]
#onnx_m.graph.output.extend(graph_outputs)

#for input_node in onnx_model.graph.input:
#	print("input name: ", input_node.name)
for output_node in onnx_model.graph.output:
	print("output name: ", output_node.name)

#print(graph.input)
print(graph.output)

input_data = np.fromfile("input.bin", dtype=np.uint8).astype(np.float32).reshape((1,3,32,256))  ## TODO for input.bin, np.uint8, shape
input_data = input_data/256.
input_fd = {'input' : input_data}	## TODO for input

sess = rt.InferenceSession(onnx_model.SerializeToString())
outs = sess.run(None, input_fd)

for arr,t in zip(outs, sess.get_outputs()):
	print(" ==> Gen Onnx PC output: ", t.name)
	arr.tofile("PC_{}.bin".format(t.name))

