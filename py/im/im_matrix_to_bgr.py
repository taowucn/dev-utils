#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import cv2

def matrix_cvt(args):
	im_data = np.fromfile(args.i, dtype=args.f)
	im_data = im_data.reshape(3, 3)
	im = np.zeros((3, 3), dtype=args.f)

	## convert
	im[0:] = im_data[2,:]
	im[1:] = im_data[1,:]
	im[2:] = im_data[0,:]

	im.tofile(args.o)
	print('Convert File: {} -> {}'.format(args.i, args.o))

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def init_param(args):
	parser = argparse.ArgumentParser(description="Swith RGB matrix (3, 3) to BGR matrix")
	parser.add_argument("-i", type=str, required=True, default="in.bin",
		help="input image filename, only fx8 format")
	parser.add_argument("-f", type=str, required=True, default="fp32",
		help="input binary format: " + dtype_str)
	parser.add_argument("-o", type=str, required=False,
		help="out image filename, do not imshow in live if gen output image")
	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	matrix_cvt(args)

