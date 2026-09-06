#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8|bool>"

def bin_matmul(args):
	import torch
	#from torch import nn

	data_src_a = np.fromfile(args.i1, dtype=args.f)
	data_src_b = np.fromfile(args.i2, dtype=args.f)

	data_src_a = data_src_a.astype(np.float32)
	data_src_b = data_src_b.astype(np.float32)

	if (args.q):
		data_src_a = data_src_a * pow(2, -(args.q))
		data_src_b = data_src_b * pow(2, -(args.q))

	# shape a,b
	a_shape = args.s1.split(',')
	b_shape = args.s2.split(',')
	a_N, a_C, a_H, a_W = list(map(int, a_shape))
	b_N, b_C, b_H, b_W = list(map(int, b_shape))
	print("input1 shape: (N, C, H, W) = ({:d}, {:d}, {:d}, {:d})".format(a_N, a_C, a_H, a_W))
	print("input2 shape: (N, C, H, W) = ({:d}, {:d}, {:d}, {:d})".format(b_N, b_C, b_H, b_W))

	data_src_a = data_src_a.reshape(a_N, a_C, a_H, a_W)
	data_src_b = data_src_b.reshape(b_N, b_C, b_H, b_W)

	# to pytorch tensor
	src_tensor_a = torch.from_numpy(data_src_a)
	src_tensor_b = torch.from_numpy(data_src_b)

	# tranpose b
	if (args.t2):
		src_tensor_b = torch.transpose(src_tensor_b, 2, 3)
		print("input2 transpose 2, 3")

	# matmul
	dst_tensor = torch.matmul(src_tensor_a, src_tensor_b)

	data_dst = dst_tensor.numpy()
	if (args.q):
		data_dst = data_dst * pow(2, (args.q))

	data_dst = data_dst.astype(args.f)
	data_dst.tofile(args.o)

	print('Shape: {} @ {} = {}'.format(src_tensor_a.shape, src_tensor_b.shape, data_dst.shape))
	print('Matmul File: {} @ {} = {}. Format: {}, Q: {}.'.format(args.i1, args.i2, args.o, args.f, args.q))


def init_param(args):
	parser = argparse.ArgumentParser(description="Compute matmul for binary file")

	parser.add_argument("-i1", type=str, required=True, default="input1.bin",
		help="input 1 filename")
	parser.add_argument("-i2", type=str, required=True, default="input2.bin",
		help="input 2 filename")

	parser.add_argument("-s1", type=str, required=True,
		help="input 1 shape. (1, 1, M, N)")
	parser.add_argument("-s2", type=str, required=True,
		help="input 2 shape. (1, 1, N, K)")

	parser.add_argument("-t2", action='store_true', required=False,
		help="input 2 if need tranpose")

	parser.add_argument("-o", type=str, required=True, default="output.bin",
		help="ouput binary filename. matmul(input1, input2) = output")

	parser.add_argument("-f", type=str, required=True, default="float32",
		help="input and output binary format: " + dtype_str)

	parser.add_argument("-q", type=int, required=False, default=0,
		help="input and output binary Q (expoffset) value ")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_matmul(args)
