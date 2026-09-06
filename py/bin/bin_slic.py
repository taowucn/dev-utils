#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def bin_slice(args):
	data = np.fromfile(args.i, dtype=args.f)
	data = data.reshape(args.n, -1)

	print("shape:", data.shape, " dtype:", data.dtype)
	h = data.shape[0]
	if (args.s > h):
		raise UserWarning("Invalid slic index: %d larger than %d" % (args.s, h))

	slic_out = data[args.s, :]
	slic_out.tofile(args.o)


def init_param(args):
	parser = argparse.ArgumentParser(description="Slice binary file with specific format, dim, slic index")
	parser.add_argument("-i", type=str, required=True, default="input.bin",
		help="input binary filename")
	parser.add_argument("-o", type=str, required=True, default="output.bin",
		help="output binary filename")
	parser.add_argument("-f", type=str, required=False, default="float32",
		help="input binary format: " + dtype_str)

	parser.add_argument("-n", type=int, required=True, default=1,
		help="slice num, view as (n, -1) uint8")
	parser.add_argument("-s", type=int, required=True, default=1,
		help="slice index, slice [slic_idx, :]")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_slice(args)
