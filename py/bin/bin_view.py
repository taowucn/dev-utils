#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def statis_data(args, data_x_f):
	if (args.s):
		max_value = np.max(data_x_f)
		min_value = np.min(data_x_f)
		mean_value = np.mean(data_x_f)
		median_value = np.median(data_x_f)
		#std_value = np.std(data_x_f)
		#var_value = np.var(data_x_f)
		diff_value = max_value - min_value
		diff_range = np.ceil(diff_value).astype(np.int32)

		print("max_value:", max_value)
		print("min_value:", min_value)
		print("diff_value:", diff_value, "diff_range:", diff_range)
		print("mean_value:", mean_value)
		print("median_value:", median_value)
		#print("std_value:", std_value)
		#print("var_value:", var_value)

def bin_view(args):
	data = np.fromfile(args.i, dtype=args.f)
	print("dtype:", args.f, ", shape:", data.shape)

	if (args.w):
		num = data.shape[0]
		height = int(num / args.w)
		print(num, " Reshape to ", height, args.w)
		data = data.reshape(height, args.w)

	if (args.q):
		data_f = data.astype(np.float32)
		data_f = data_f * pow(2, -(args.q))
		if (args.v):
			for row in data_f:
				print(row)
		else:
			print(data_f)
			statis_data(args, data_f)
	else:
		if (args.v):
			for row in data:
				print(row)
		else:
			print(data)
			statis_data(args, data)

	threshold_max = args.a
	if threshold_max:
		filter_data = np.where(data > threshold_max)
		filter_data = np.asarray(filter_data)
		print("\n---------------- show bigger than threahold.Max:", threshold_max)
		#print(filter_data)
		for idx in np.nditer(filter_data):
			print("idx: {}, data: {})".format(idx, data[idx]))
		print("data bigger than threshold.Max {} shape: {}".format(threshold_max, filter_data.shape))

	threshold_min = args.b
	if threshold_min:
		filter_data = np.where(data < threshold_min)
		filter_data = np.asarray(filter_data)
		print("\n---------------- show smaller than threahold.Min:", threshold_min)
		#print(filter_data)
		for idx in np.nditer(filter_data):
			print("idx: {}, data: {})".format(idx, data[idx]))
		print("data smaller than threshold.Min {} shape: {}".format(threshold_min, filter_data.shape))

def init_param(args):
	parser = argparse.ArgumentParser(description="View binary file with specific format")
	parser.add_argument("-i", type=str, required=True, default="input.bin",
		help="input binary filename")
	parser.add_argument("-f", type=str, required=False, default="float32",
		help="input binary format: " + dtype_str)
	parser.add_argument("-q", type=int, required=False,
		help="Q value for quantized data")
	parser.add_argument("-w", type=int, required=False,
		help="Show element number in row (width)")
	parser.add_argument("-v", action='store_true', required=False,
		help="Show data by print txt")

	parser.add_argument("-a", type=float, required=False,
		help="threshold.Max, show data when data > threshold.Max")
	parser.add_argument("-b", type=float, required=False,
		help="threshold.Min, show data when data < threshold.Min")

	parser.add_argument("-s", action='store_true', required=False, default=False,
		help="statis data (max, min, avg, std etc.)")

	parser.add_argument("-W", type=int, required=False, default=10,
		help="image width")
	parser.add_argument("-H", type=int, required=False, default=1,
		help="image height")
	parser.add_argument("-C", type=int, required=False, default=1,
		help="image channel")
	parser.add_argument("-coord", type=str, required=False,
		help="coordinate, format(x, y) at (w, h)")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_view(args)
