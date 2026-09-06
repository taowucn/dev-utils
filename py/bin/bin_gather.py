#!/usr/bin/env python3

import os, sys, argparse
import numpy as np


dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8|bool>"

def bin_gather(args):
	data_table = np.fromfile(args.i, dtype=args.f)
	data_table = data_table.reshape(-1, args.s)

	data_gather_idx = np.fromfile(args.g, dtype=args.fg)
	data_gather_idx = data_gather_idx.reshape(-1)

	print("input table shape:", data_table.shape)
	print("input gather index shape:", data_gather_idx.shape)

	if (args.v):
		print("input table:", data_table)
		print("input gather index:", data_gather_idx)

	data_dst = data_table[data_gather_idx]
	data_dst.tofile(args.o)
	print("gather File: {} shape: {}".format(args.o, data_dst.shape))

def init_param(args):
	parser = argparse.ArgumentParser(description="Gather table by index")
	parser.add_argument("-i", type=str, required=True, default="input.bin",
		help="input gather table filename")
	parser.add_argument("-g", type=str, required=True, default="gather_idx.bin",
		help="input gather index filename")

	parser.add_argument("-o", type=str, required=True, default="output.bin",
		help="ouput binary filename")

	parser.add_argument("-s", type=int, required=True,
		help="table low dim size, reshape to (-1, size)")

	parser.add_argument("-f", type=str, required=True, default="float32",
		help="input and output binary format: " + dtype_str)

	parser.add_argument("-fg", type=str, required=True, default="int32",
		help="gather index binary format: " + dtype_str)

	parser.add_argument("-v", action='store_true', required=False,
		help="Show data")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_gather(args)
