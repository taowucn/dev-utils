#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import onnx
import onnxruntime as rt
from onnx import helper

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

new_outs = ['latent_sample']

def append_new_outs(onnx_m):
	print("Append new output name: ", new_outs)

	graph_outputs = [helper.make_empty_tensor_value_info(t) for t in new_outs]
	#+ list(onnx_m.graph.output)
	#del onnx_m.graph.output[:]
	onnx_m.graph.output.extend(graph_outputs)


def inference_model(args):
	onnx_model = onnx.load(args.model)
	graph = onnx_model.graph

	# if append intermediate outputs:
	if (args.append):
		append_new_outs(onnx_model)
	#print(graph)

	#for input_node in onnx_model.graph.input:
	#	print("input name: ", input_node.name)
	for output_node in onnx_model.graph.output:
		print("output name: ", output_node.name)

	if (args.v):
		#print(graph.input)
		print(graph.output)

	# FIXME: fixed input shape
	input_data = np.fromfile(args.bin, dtype=args.fmt).astype(np.float32).reshape((1,3,1024,1024))
	# normalize data
	input_data = input_data/256.
	input_fd = {args.name : input_data}

	sess = rt.InferenceSession(onnx_model.SerializeToString())
	outs = sess.run(None, input_fd)

	for arr,t in zip(outs, sess.get_outputs()):
		print(" ==> Gen Onnx PC output: ", t.name)
		arr.tofile("PC_{}.bin".format(t.name))

def init_param(args):
	parser = argparse.ArgumentParser(description="Inference Onnx Model by feeding one input file")

	parser.add_argument("-model", type=str, required=True, default="model.onnx",
		help="onnx model file")

	parser.add_argument("-name", type=str, required=True, default="data",
		help="input name")
	parser.add_argument("-bin", type=str, required=True, default="input.bin",
		help="input binary file")

	parser.add_argument("-fmt", type=str, required=False, default="uint8",
		help="input binary format: " + dtype_str)

	parser.add_argument("-append", action='store_true', required=False,
		help="Append new intermediate outputs")

	parser.add_argument("-v", action='store_true', required=False,
		help="Show verbose info")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	inference_model(args)
