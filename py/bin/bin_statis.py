#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def statis_data(data_x_f):
	max_value = np.max(data_x_f)
	min_value = np.min(data_x_f)
	mean_value = np.mean(data_x_f)
	median_value = np.median(data_x_f)
	#std_value = np.std(data_x_f)
	#var_value = np.var(data_x_f)
	diff_value = max_value - min_value
	diff_range = np.ceil(diff_value).astype(np.int32)
	sum_value = np.sum(data_x_f)

	print("max_value:", max_value)
	print("min_value:", min_value)
	print("diff_value:", diff_value, "diff_range:", diff_range)
	print("mean_value:", mean_value)
	print("median_value:", median_value)
	#print("std_value:", std_value)
	#print("var_value:", var_value)
	print("sum_value:", sum_value)

	hist, bin_edges = np.histogram(data_x_f, bins='auto', density=False)
	print("hist:", hist)
	print("bin_edges:", bin_edges)

def bin_view(args):
	data = np.fromfile(args.i, dtype=args.f)
	print("dtype:", args.f, ", shape:", data.shape)

	data = data.astype(np.float32)
	if (args.q):
		data = data * pow(2, -(args.q))

	#print(data)
	statis_data(data)


def init_param(args):
	parser = argparse.ArgumentParser(description="Statis binary file with specific format")
	parser.add_argument("-i", type=str, required=True, default="input.bin",
		help="input binary filename")
	parser.add_argument("-f", type=str, required=False, default="float32",
		help="input binary format: " + dtype_str)
	parser.add_argument("-q", type=int, required=False,
		help="Q value for quantized data")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_view(args)
