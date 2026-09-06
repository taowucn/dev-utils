#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8|bool>"

def bin_add(args):
	data_src1 = np.fromfile(args.i1, dtype=args.f)
	data_src1 = data_src1.astype(np.float32)

	data_src2 = np.fromfile(args.i2, dtype=args.f)
	data_src2 = data_src2.astype(np.float32)

	if (args.q):
		data_src1 = data_src1 * pow(2, -(args.q))
		data_src2 = data_src2 * pow(2, -(args.q))

	data_dst = data_src1 + data_src2
	if (args.v):
		print("input1:", data_src1)
		print("input2:", data_src2)
		print("output:", data_dst)

	if (args.q):
		data_dst = data_dst * pow(2, (args.q))

	data_dst = data_dst.astype(args.f)
	data_dst.tofile(args.o)

	print('Add two files: {} + {} = {}; Format: {}; Q: {}; out.shape:{} '.format(args.i1, args.i2, args.o, args.f, args.q, data_dst.shape))

def init_param(args):
	parser = argparse.ArgumentParser(description="Compuate add for two binary file")
	parser.add_argument("-i1", type=str, required=True, default="input1.bin",
		help="input image filename")
	parser.add_argument("-i2", type=str, required=True, default="input2.bin",
		help="input image filename")

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
	bin_add(args)
