#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def bin_to_img(args):
	data = np.fromfile(args.i, dtype=args.f)
	print("dtype:", args.f)

	# shape a
	a_shape = args.s.split(',')
	a_C, a_H, a_W = list(map(int, a_shape))
	print("input shape: (C, H, W) = ({:d}, {:d}, {:d})".format(a_C, a_H, a_W))
	data = data.reshape(a_C, a_H, a_W)
	data = data.transpose(1, 2, 0)

	data_f = data
	if (args.q):
		data_f = data.astype(np.float32)
		data_f = data_f * pow(2, -(args.q))

	plt.imsave(args.o, data_f)
	print("Save image:", args.o, "Done")

def init_param(args):
	parser = argparse.ArgumentParser(description="View binary file with specific format")
	parser.add_argument("-i", type=str, required=True, default="input.bin",
		help="input binary filename")
	parser.add_argument("-f", type=str, required=False, default="float32",
		help="input binary format: " + dtype_str)
	parser.add_argument("-q", type=int, required=False,
		help="Q value for quantized data")

	parser.add_argument("-s", type=str, required=True,
		help="input binary shape. (C, H, W)")

	parser.add_argument("-o", type=str, required=True,
		help="output image. e.g. img.jpg")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_to_img(args)
