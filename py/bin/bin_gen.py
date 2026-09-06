#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def bin_gen(args):
	if (args.c):
		data = np.arange(args.n, dtype=args.f)
	elif (args.a):
		value = float(args.a)
		data = np.linspace(-value, value, args.n)
	elif (args.s):
		data = np.random.randn(args.n)
	elif (args.d):
		value = float(args.d)
		data = np.ones(args.n, dtype=args.f) * value

	if (args.q):
		data = data * pow(2, (args.q))

	data = data.astype(dtype=args.f)
	data.tofile(args.o)
	filesize=os.path.getsize(args.o)

	if (args.c):
		print('Generate File: {}, Data: {}, Num: {}, Size: {} byte'.format(args.o, "0~N", args.n, filesize))
	elif (args.a):
		print('Generate File: {}, Linsapce: [ {}, {} ], Num: {}, Size: {} byte'.format(args.o, -value, value, args.n, filesize))
	elif (args.s):
		print('Generate File: {}, Standard normal: mean=0, sigma=1, Num: {}, Size: {} byte'.format(args.o, args.n, filesize))
	elif (args.d):
		print('Generate File: {}, Data: {}, Num: {}, Size: {} byte'.format(args.o, value, args.n, filesize))

def init_param(args):
	parser = argparse.ArgumentParser(description="Generate binary file with specific format and data")
	parser.add_argument("-o", type=str, required=True, default="output.bin",
		help="ouput binary filename")
	parser.add_argument("-f", type=str, required=True, default="float32",
		help="binary format: " + dtype_str)
	parser.add_argument("-q", type=int, required=False,
		help="Q value for quantized data")
	parser.add_argument("-d", type=str, required=False, default="1",
		help="binary data value")
	parser.add_argument("-c", action="store_true", default=False,
		help="binary data value from 0 to N increase with 1")
	parser.add_argument("-a", type=int, default=None,
		help="binary data value linspace in range [-value, value]")
	parser.add_argument("-s", action="store_true", default=None,
		help="binary data value standard normal, mean=0, sigma=1")
	parser.add_argument("-n", type=int, required=True, default=1,
		help="element number")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_gen(args)
