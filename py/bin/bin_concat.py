#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8|bool>"

def bin_concat(args):
	data_src_a = np.fromfile(args.i1, dtype=args.f)
	data_src_b = np.fromfile(args.i2, dtype=args.f)

	# shape
	a_shape = args.s.split(',')
	a_N, a_C, a_H, a_W = list(map(int, a_shape))
	print("input shape: (N, C, H, W) = ({:d}, {:d}, {:d}, {:d})".format(a_N, a_C, a_H, a_W))

	data_src_a = data_src_a.reshape(a_N, a_C, a_H, a_W)
	data_src_b = data_src_b.reshape(a_N, a_C, a_H, a_W)

	if (args.d):
		axis_dim = args.d
	else:
		axis_dim = -1
	print("concat axis: ", axis_dim)

	# concat
	dst_data = np.concatenate((data_src_a, data_src_b), axis=axis_dim)
	dst_data.tofile(args.o)

	print('Output File: {}  shape: {}'.format(args.o, dst_data.shape))


def init_param(args):
	parser = argparse.ArgumentParser(description="Concat two binary file into one file with specify dim")

	parser.add_argument("-i1", type=str, required=True, default="input1.bin",
		help="input 1 filename")
	parser.add_argument("-i2", type=str, required=True, default="input2.bin",
		help="input 2 filename")

	parser.add_argument("-o", type=str, required=True, default="output.bin",
		help="ouput binary filename. matmul(input1, input2) = output")

	parser.add_argument("-f", type=str, required=True, default="int8",
		help="input and output binary format: " + dtype_str)

	parser.add_argument("-s", type=str, required=True,
		help="two input shape. (N, C, H, W)")
	parser.add_argument("-d", type=int, required=False, default=0,
		help="concat axis (0, 1, 2, 3) or (-4, -3,-2,-1)")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_concat(args)
