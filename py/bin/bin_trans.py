#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

def transform(args):
	data = np.fromfile(args.i, dtype=args.f)
	a_shape = args.s.split(',')
	b_shape = args.t.split(',')
	a_N, a_C, a_H, a_W = list(map(int, a_shape))
	b_N, b_C, b_H, b_W = list(map(int, b_shape))

	data = data.reshape(a_N, a_C, a_H, a_W)
	print("input shape: ", data.shape)
	print("transpose parmas: ({:d}, {:d}, {:d}, {:d})".format(b_N, b_C, b_H, b_W))

	dst = data.transpose(b_N, b_C, b_H, b_W)
	print("output shape:", dst.shape)

	dst.tofile(args.o)
	print('Transpose File: {} -> {}'.format(args.i, args.o))

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def init_param(args):
	parser = argparse.ArgumentParser(description="Transpose binary file")
	parser.add_argument("-i", type=str, required=True, default="input.bin",
		help="input image filename")
	parser.add_argument("-o", type=str, required=True, default="output.bin",
		help="ouput binary filename")
	parser.add_argument("-f", type=str, required=True, default="float32",
		help="dest binary format: " + dtype_str)

	parser.add_argument("-s", type=str, required=True,
		help="input shape. (N, C, H, W)")
	parser.add_argument("-t", type=str, required=True,
		help="transpose params. (0, 1, 2, 3)")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	transform(args)

