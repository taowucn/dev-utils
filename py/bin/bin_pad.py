#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

def bin_pad(args):
	data = np.fromfile(args.i, dtype=args.f)
	a_shape = args.s.split(',')
	b_shape = args.t.split(',')
	a_N, a_C, a_H, a_W = list(map(int, a_shape))
	b_N, b_C, b_H, b_W = list(map(int, b_shape))
	# c_to_w_interlave = arg.c

	print("orginal shape:", a_N, b_C, a_H, a_W)
	print("padded shape:", b_N, b_C, b_H, b_W)
	print("-"*60)

	if args.a:
		## input data
		data = data.reshape(a_N, a_C, a_H, a_W)
		print("input shape: ", data.shape)

		## add pad in here
		dst = np.zeros((b_N, b_C, b_H, b_W), dtype=data.dtype)
		dst[:a_N, :a_C, :a_H, :a_W] = data
		print("output shape:", dst.shape)

		if args.c:
			raise "Not verify"


			## C-to-W interleave DRAM layout: fold C into W.
			## (N, C, H, W) -> (N, 1, H, C*W), each channel's full row
			## laid out contiguously so k = c * W + w.
			dst = dst.transpose(0, 2, 1, 3).reshape(b_N, 1, b_H, b_C * b_W)
			dst = np.ascontiguousarray(dst)
			print("interleave shape:", dst.shape)

		dst.tofile(args.o)
		print('Add Pad File: {} -> {}'.format(args.i, args.o))
	else:
		## input data
		if args.c > 1:
			raise "Not verify"
			## un-fold C-to-W interleave DRAM layout back to (N, C, H, W)
			n_H = b_H / c_to_w_interlave
			data = data.reshape(b_N, n_H, b_C, b_W, c_to_w_interlave).transpose(0, 2, 1, 3)
			data = np.ascontiguousarray(data)
		else:
			data = data.reshape(b_N, b_C, b_H, b_W)
		print("input shape: ", data.shape)

		## remove pad
		dst = data[:a_N, :a_C, :a_H, :a_W]
		print("output shape:", dst.shape)

		dst.tofile(args.o)
		print('Remove Pad File: {} -> {}'.format(args.i, args.o))


dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def init_param(args):
	parser = argparse.ArgumentParser(description="Remove padding data in binary file")
	parser.add_argument("-i", type=str, required=True, default="input.bin",
		help="input image filename")
	parser.add_argument("-o", type=str, required=True, default="output.bin",
		help="ouput binary filename")
	parser.add_argument("-f", type=str, required=True, default="float32",
		help="dest binary format: " + dtype_str)

	parser.add_argument("-a", action='store_true', required=False, default=False,
		help="Add padding data (zero), default is remove padding data")

	parser.add_argument("-c", type=int, required=False, default=1,
		help="c_to_w_interlave")

	parser.add_argument("-s", type=str, required=True,
		help="orginal shape. (N, C, H, W)")
	parser.add_argument("-t", type=str, required=True,
		help="padded shape. (N_pad, C_pad, H_pad, W_pad)")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_pad(args)

