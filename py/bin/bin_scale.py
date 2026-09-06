#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8|bool>"

def bin_scale(args):
	data_src = np.fromfile(args.i, dtype=args.f)
	data_src = data_src.astype(np.float32)

	if (args.q):
		data_src = data_src * pow(2, -(args.q))

	data_dst = data_src * args.s
	if (args.v):
		print("input:", data_src)
		print("scale factor:", args.s)
		print("output:", data_dst)

	if (args.q):
		data_dst = data_dst * pow(2, (args.q))

	data_dst = data_dst.astype(args.f)
	data_dst.tofile(args.o)

	print('Scale file: {} * {} = {}; Format: {}; Q: {}; out.shape:{} '.format(args.i, args.s, args.o, args.f, args.q, data_dst.shape))

def init_param(args):
	parser = argparse.ArgumentParser(description="Scale one binary file by a scale factor")
	parser.add_argument("-i", type=str, required=True, default="input1.bin",
		help="input image filename")
	parser.add_argument("-s", type=float, required=True, default=1.0,
		help="scale factor. e.g. 1.0, 0.5, 2.0, etc.")

	parser.add_argument("-o", type=str, required=True, default="output.bin",
		help="ouput binary filename")

	parser.add_argument("-f", type=str, required=True, default="float32",
		help="input and output binary format: " + dtype_str)

	parser.add_argument("-q", type=int, required=False, default=0,
		help="input and output binary Q (expoffset) value ")

	parser.add_argument("-v", action='store_true', required=False,
		help="Show data")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_scale(args)
