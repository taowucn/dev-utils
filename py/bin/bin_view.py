#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def bin_view(args):
	data = np.fromfile(args.i, dtype=args.f)

	if (args.w):
		num = data.shape[0]
		height = int(num / args.w)
		print(num, " Reshape to ", height, args.w)
		data = data.reshape(height, args.w)

	if (args.q):
		data_f = data.astype(np.float32)
		data_f = data_f/pow(2, args.q)
		if (args.v):
			for row in data_f:
				print(row)
		else:
			print(data_f)
	else:
		if (args.v):
			for row in data:
				print(row)
		else:
			print(data)

	threshold = args.t
	if (threshold > 0):
		filter_data = np.where(data > threshold)
		filter_data = np.asarray(filter_data)
		print("show bigger than threahold:", threshold)
		#print(filter_data)
		for idx in np.nditer(filter_data):
			print("idx: {}, data: {})".format(idx, data[idx]))
		print("data bigger than threshold {} shape: {}".format(threshold, filter_data.shape))


def init_param(args):
	parser = argparse.ArgumentParser(description="View binary file with specific format")
	parser.add_argument("-i", type=str, required=True, default="input.bin",
		help="input binary filename")
	parser.add_argument("-f", type=str, required=False, default="uint8",
		help="input binary format: " + dtype_str)
	parser.add_argument("-q", type=int, required=False,
		help="Q value for quantized data")
	parser.add_argument("-w", type=int, required=False,
		help="Show element number in row (width)")
	parser.add_argument("-v", action='store_true', required=False,
		help="Show data by print txt")

	parser.add_argument("-t", type=int, required=False, default=0,
		help="threshold, show data when data > threshold")

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
