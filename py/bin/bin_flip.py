#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8|bool>"

def bin_flip(args):
	data_src = np.fromfile(args.i, dtype=args.f)

	# shape
	a_shape = args.s.split(',')
	a_N, a_C, a_H, a_W = list(map(int, a_shape))
	print("input shape: (N, C, H, W) = ({:d}, {:d}, {:d}, {:d})".format(a_N, a_C, a_H, a_W))
	data_src = data_src.reshape(a_N, a_C, a_H, a_W)

	if (args.d):
		axis_dim = args.d
	else:
		axis_dim = -1
	print("flip axis: ", axis_dim)

	data_dst = np.flip(data_src, axis=args.d)
	data_dst.tofile(args.o)
	print('Output File: {}  shape: {}'.format(args.o, data_dst.shape))

def init_param(args):
	parser = argparse.ArgumentParser(description="Flip binary file")
	parser.add_argument("-i", type=str, required=True, default="input.bin",
		help="input image filename")
	parser.add_argument("-o", type=str, required=True, default="output.bin",
		help="ouput binary filename")
	parser.add_argument("-f", type=str, required=True, default="int8",
		help="input binary format: " + dtype_str)
	parser.add_argument("-s", type=str, required=True,
		help="input shape. (N, C, H, W)")
	parser.add_argument("-d", type=int, required=False, default=0,
		help="flip axis (0, 1, 2, 3) or (-4, -3,-2,-1)")
	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_flip(args)
