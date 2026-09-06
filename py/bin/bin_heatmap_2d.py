#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def bin_heatmap(args):
	data_x = np.fromfile(args.i, dtype=args.f)
	data_x_f = data_x.astype(np.float32)

	if (args.q):
		data_x_f = data_x_f/pow(2, args.q)

	if (args.v):
		print("--- x data in float32 ---")
		print("x:", data_x_f)

	max_value = np.max(data_x_f)
	min_value = np.min(data_x_f)
	mean_value = np.mean(data_x_f)
	median_value = np.median(data_x_f)
	std_value = np.std(data_x_f)
	var_value = np.var(data_x_f)
	diff_value = max_value - min_value
	diff_range = np.ceil(diff_value).astype(np.int32)

	if (args.s):
		print("max_value:", max_value)
		print("min_value:", min_value)
		print("diff_value:", diff_value, "diff_range:", diff_range)
		print("mean_value:", mean_value)
		print("median_value:", median_value)
		print("std_value:", std_value)
		print("var_value:", var_value)

	shape_arg = args.shape.split(',')
	height, width = list(map(int, shape_arg))

	data_x_f = data_x_f.reshape(height, width)
	print("Shape: (h, w) = ({:d}, {:d})".format(height, width), "shape: ", data_x_f.shape)

	hist = plt.imshow(data_x_f)

	# show result
	if (args.m != 'numpy'):
		if (args.o):
			print("save image as:", args.o)
			plt.imsave(args.o, hist)
		else:
			print("show image in live")
			plt.show()

def init_param(args):
	parser = argparse.ArgumentParser(description="Show binary file in heatmap 2D with specific format, 1.0.0")
	parser.add_argument("-i", type=str, required=True, default="in.bin",
		help="input binary filename")
	parser.add_argument("-f", type=str, required=False, default="float32",
		help="input binary format: " + dtype_str)
	parser.add_argument("-q", type=int, required=False,
		help="Q value for quantized data")

	parser.add_argument("-s", action='store_true', required=False, default=True,
		help="statis data (max, min, avg, std etc.)")

	parser.add_argument("-shape", type=str, required=True,
		help="shape, format(h, w)")
	parser.add_argument("-v", action='store_true', required=False,
		help="verbose log like show data")

	parser.add_argument("-o", type=str, required=False,
		help="out image filename (jpg, png), do not imshow in live if gen output image")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_heatmap(args)

