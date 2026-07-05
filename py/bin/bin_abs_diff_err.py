#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def get_coord(idx, c, h, w):
	line = idx
	one_c_size = (h * w)

	dc = int(line/one_c_size)
	ret = line - (dc * one_c_size)
	dh = int(ret/w)
	dw = ret - (dh * w)

	return (dc, dh, dw)

def abs_diff_err(args):
	data_left = np.fromfile(args.f1, dtype=args.f)
	data_right = np.fromfile(args.f2, dtype=args.f)
	data_left = data_left.astype(np.float32)
	data_right = data_right.astype(np.float32)
	abs_diff = np.abs(data_left - data_right)
	max_diff = abs_diff.max()
	idx = np.argmax(abs_diff)

	width = args.W
	height = args.H
	chan = args.C

	print("--------------", args.f1, "--------------")
	print(data_left)
	print("--------------", args.f2, "--------------")
	print(data_right)
	print("============")
	dc, dh, dw = get_coord(idx, chan, height, width)
	print("max_diff: {}, idx: {}, coord (C, H, W): ({}, {}, {})".format(max_diff, idx, dc, dh, dw))

	filter_abs_diff = np.where(abs_diff > args.t)
	filter_abs_diff = np.asarray(filter_abs_diff)

	count = 0
	for idx in np.nditer(filter_abs_diff):
		if ((args.n > 0) & (count > args.n)):
			print("Only show head number", args.n)
			break
		count+=1
		dc, dh, dw = get_coord(idx, chan, height, width)
		print("abs_diff: {} = {} - {}, idx: {}, coord (C, H, W): ({}, {}, {})".format(abs_diff[idx], data_left[idx], data_right[idx], idx, dc, dh, dw))

	print("abs.diff bigger than threshold {} shape: {}".format(args.t, filter_abs_diff.shape))


def init_param(args):
	parser = argparse.ArgumentParser(description="Calculate abs.diff err with two files, and show detailed coordinate of difference")
	parser.add_argument("-f1", type=str, required=True, default="a.bin",
		help="left input binary filename")
	parser.add_argument("-f2", type=str, required=True, default="b.bin",
		help="right input binary filename")
	parser.add_argument("-f", type=str, required=True, default="fp32",
		help="input binary format: " + dtype_str)

	parser.add_argument("-W", type=int, required=False, default=10,
		help="image width")
	parser.add_argument("-H", type=int, required=False, default=1,
		help="image height")
	parser.add_argument("-C", type=int, required=False, default=1,
		help="image channel")
	parser.add_argument("-t", type=int, required=False, default=64,
		help="threshold, show data when abs.diff > threshold")
	parser.add_argument("-n", type=int, required=False, default=0,
		help="show head number only")
	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	abs_diff_err(args)
