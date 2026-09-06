#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import math

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8|bool>"

def bin_rmsnorm(args):
	data_src = np.fromfile(args.i, dtype=args.f)
	data_src = data_src.astype(np.float32)

	data_len = len(data_src)
	print("len:", data_len)

	if (args.q):
		data_src = data_src * pow(2, -(args.q))

	data_pow = np.multiply(data_src, data_src)
	sumed = np.sum(data_pow)
	sum_scaled = sumed / data_len
	sum_scaled = sum_scaled + args.e
	ss = 1.0 / math.sqrt(sum_scaled)
	data_dst = data_src * ss

	if (args.w):
		data_wte = np.fromfile(args.w, dtype=args.f)
		data_wte = data_wte.astype(np.float32)
		data_dst = np.multiply(data_dst, data_wte)

	if (args.v):
		print("input:", data_src)
		print("output:", data_dst)

	if (args.q):
		data_dst = data_dst * pow(2, (args.q))

	data_dst = data_dst.astype(args.f)
	data_dst.tofile(args.o)

	print('RMSNorm without weight : {} -> {}; len: {}, epson: {}, sum+eps: {} scale: {} '.format(args.i, args.o, data_len, args.e, sumed, ss))

def init_param(args):
	parser = argparse.ArgumentParser(description="Compuate rmsnorm with weight scale for binary file")
	parser.add_argument("-i", type=str, required=True, default="input1.bin",
		help="input image filename")

	parser.add_argument("-w", type=str, required=True, default="wte.bin",
		help="rmsnorm weight filename, use 1.0 if not specify it.")

	parser.add_argument("-o", type=str, required=True, default="output.bin",
		help="ouput binary filename")

	parser.add_argument("-f", type=str, required=True, default="float32",
		help="input and output binary format: " + dtype_str)

	parser.add_argument("-e", type=float, required=False, default=1.0e-5,
		help="epson. e.g. 1.0e-5 ")

	parser.add_argument("-q", type=int, required=False, default=0,
		help="input and output binary Q (expoffset) value ")

	parser.add_argument("-v", action='store_true', required=False,
		help="Show data by print txt")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_rmsnorm(args)
