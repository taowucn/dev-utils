#!/usr/bin/env python3

import os, sys, argparse
#import numpy as np
import pickle

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def pkl_view(args):
	with open(args.i, 'rb') as file:
		data = pickle.load(file)

	print(data)

def init_param(args):
	parser = argparse.ArgumentParser(description="View pickle file")
	parser.add_argument("-i", type=str, required=True, default="input.pkl",
		help="input *.pkl filename")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	pkl_view(args)
