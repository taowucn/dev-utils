#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import onnx
import onnxruntime as rt
from onnx import helper

def run_onnx_mode(onnx_model_file, input_bin_file):
	print("ONNX Model:", onnx_model_file)
	print("Input bin file:", input_bin_file)

	onnx_model = onnx.load(onnx_model_file)  ## TODO for sample.onnx
	graph = onnx_model.graph

	#new_outs = ['output']  ## TODO for intermediate data
	#append intermediate outputs:
	#graph_outputs = [helper.make_empty_tensor_value_info(t) for t in new_outs]
	#onnx_m.graph.output.extend(graph_outputs)

	for input_node in onnx_model.graph.input:
		print("input name: ", input_node.name)
	for output_node in onnx_model.graph.output:
		print("output name: ", output_node.name)

	#print(graph.input)
	#print(graph.output)

	input_data = np.fromfile(input_bin_file, dtype=np.uint8).astype(np.float32).reshape((1, 1, 28, 28))  ## TODO for input.bin, np.uint8, shape
	#input_data = input_data/256.
	input_fd = {'input' : input_data}	## TODO for input

	sess = rt.InferenceSession(onnx_model.SerializeToString())
	outs = sess.run(None, input_fd)

	for arr, t in zip(outs, sess.get_outputs()):
		out_fn = "out_onnx_{}.bin".format(t.name)
		print("output name: ", t.name, ", data:", arr)
		arr.tofile(out_fn)
		print("  save to bin file:", out_fn)

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print("Usage: " + sys.argv[0] + " <ONNX Model> <input bin>")
	else:
		run_onnx_mode(sys.argv[1], sys.argv[2])
