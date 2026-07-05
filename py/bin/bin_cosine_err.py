#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def calculate_cosine(args):
	data_left = np.fromfile(args.f1, dtype=args.f)
	data_right = np.fromfile(args.f2, dtype=args.f)
	cosine = np.dot(data_left, data_right) / (np.linalg.norm(data_left) * np.linalg.norm(data_right))

	print("--------------", args.f1)
	print(data_left)
	print("--------------", args.f2)
	print(data_right)
	print("============")
	print("cosine_dist:", cosine)

def init_param(args):
	parser = argparse.ArgumentParser(description="Calculate cosin distance with two files")
	parser.add_argument("-f1", type=str, required=True, default="left.bin",
		help="left input binary filename")
	parser.add_argument("-f2", type=str, required=True, default="right.bin",
		help="right input binary filename")
	parser.add_argument("-f", type=str, required=True, default="fp32",
		help="input binary format: " + dtype_str)
	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	calculate_cosine(args)
