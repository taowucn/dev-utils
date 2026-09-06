#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

from math import nan
from scipy import linalg
from scipy import spatial as sp

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def check_fsize(f1, f2):
	if (len(f1) != len(f2)):
		print("Two file size: ", len(f1), len(f2))
		return False
	else:
		return True

def relative_error(f1, f2):
	diff = f1 - f2
	#relative_err = 100*np.where(np.isclose(f1, 0, rtol=0.0001), f2, np.absolute(np.divide(diff, f1)))
	#relative_err = np.where(np.isclose(f1, 0, rtol=0.0001), f2, np.absolute(np.divide(diff, f1)))
	#is_close = np.isclose(f1, f2, rtol=1.e-3, atol=1.e-3, equal_nan=False)
	all_close = np.allclose(f1, f2, rtol=1.e-3, atol=1.e-3, equal_nan=False)

	return all_close

def bin_err(args):
	data_a = np.fromfile(args.f1, dtype=args.t1)
	data_b = np.fromfile(args.f2, dtype=args.t2)

	if (data_a.dtype) == (data_b.dtype):
		print("Two files use the same dtype: ", args.t1, args.t2)
		if not check_fsize(data_a, data_b):
			print("Two files size different, error")
			return
	else:
		print("Two files use different dtype: ", args.t1, args.t2)

	data_fa = data_a.astype(np.float32)
	data_fb = data_b.astype(np.float32)

	if (args.q1):
		data_fa = data_fa/pow(2, args.q1)
	if (args.q2):
		data_fb = data_fb/pow(2, args.q2)

	if (args.v):
		print("--------------", args.f1)
		print(data_fa)
		print("--------------", args.f2)
		print(data_fb)
		print("============")

	all_close = relative_error(data_fa, data_fb)
	print("all_close:", all_close)

	if not all_close:
		print("++++ not_close ++++")
		return 1
		#raise UserWarning("not close")

def init_param(args):
	parser = argparse.ArgumentParser(description="Compare two binary file if is close (by np.allclose)")
	parser.add_argument("-f1", type=str, required=True, default="a.bin",
		help="input binary 1 filename")
	parser.add_argument("-f2", type=str, required=True, default="b.bin",
		help="input binary 2 filename")
	parser.add_argument("-t1", type=str, required=False, default="float32",
		help="input binary 1 format: " + dtype_str)
	parser.add_argument("-t2", type=str, required=False, default="float32",
		help="input binary 2 format: " + dtype_str)
	parser.add_argument("-q1", type=int, required=False,
		help="Q value for quantized data 1")
	parser.add_argument("-q2", type=int, required=False,
		help="Q value for quantized data 2")
	parser.add_argument("-v", action='store_true', required=False,
		help="Show data in float32")
	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_err(args)
