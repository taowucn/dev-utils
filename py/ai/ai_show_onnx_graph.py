#!/usr/bin/env python3
import os, sys, argparse
import onnx
from onnx import numpy_helper

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def shape2tuple(shape):
	return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)

def show_model(args):
	# Load the ONNX model
	print("Model: ", args.model)
	onnx_model = onnx.load(args.model)
	graph = onnx_model.graph
	node = graph.node

	# Check that the model is well formed
	if args.check:
		onnx.checker.check_model(onnx_model)

	print("Node num:", len(node))

	if args.d == 1:
		for i in range(len(node)):
			print("name:", node[i].name, " op_type:", node[i].op_type)
	elif args.d == 2:
		# Print a human readable representation of the graph
		print(onnx.helper.printable_graph(onnx_model.graph))

	print("\n------ Model Primary IO  ------")
	print("Input num:", len(onnx_model.graph.input))
	for obj in onnx_model.graph.input:
		print("  input:", obj.name, shape2tuple(obj.type.tensor_type.shape))

	print("Output num:", len(onnx_model.graph.output))
	for obj in onnx_model.graph.output:
		print("  output:", obj.name, shape2tuple(obj.type.tensor_type.shape))

	if args.exportw:
		out_folder = args.exportw
		os.makedirs(out_folder, exist_ok=True)
		print("\n------ Model Export Weights  ------")
		print("export-weights out folder:", out_folder)
		if args.f:
			print("export-weights in format: ", args.f)

		for t in onnx_model.graph.initializer:
			out_fn = os.path.join(out_folder, t.name)
			w = numpy_helper.to_array(t)
			if args.d > 0:
				print(t.name)
			if args.f:
				w = w.astype(dtype=args.f)
			## save
			w.tofile(out_fn)


def init_param(args):
	parser = argparse.ArgumentParser(description="Show Onnx Model")

	parser.add_argument("-model", type=str, required=True, default="model.onnx",
		help=".onnx model file")

	parser.add_argument("-check", type=int, required=False, default=0,
		help="check_model")

	parser.add_argument("-d", type=int, required=False, default=0,
		help="Show log level info. [0, 2]")

	parser.add_argument("-exportw", type=str, required=False,
		help="export weights to output folder")
	parser.add_argument("-f", type=str, required=False,
		help="export weights with data format: " + dtype_str + ", if not specified, use the default df in original model")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	show_model(args)
