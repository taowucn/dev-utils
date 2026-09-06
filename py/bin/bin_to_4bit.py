#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def bin_convert_to_4bit(args):
	data = np.fromfile(args.i, dtype=args.f)
	print("dtype:", args.f, ", shape:", data.shape)

	w4 = data.reshape(-1, 2).transpose(1, 0).astype(np.uint8)
	nw = np.bitwise_and(w4[0], 15) + np.bitwise_and(w4[1], 15) * 16

	nw.tofile(args.o)


def init_param(args):
	parser = argparse.ArgumentParser(description="Convert other format container data to uint4_t")
	parser.add_argument("-i", type=str, required=True, default="input.bin",
		help="input binary filename")

	parser.add_argument("-f", type=str, required=True, default="uint8",
		help="input binary container format: " + dtype_str + ", but 4bit value)")

	parser.add_argument("-o", type=str, required=True, default="output.bin",
		help="output uint4_t binary filename")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_convert_to_4bit(args)
