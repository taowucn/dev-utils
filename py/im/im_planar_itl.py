#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import cv2

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def cvt_planar_itl(args):
	im_data = np.fromfile(args.i, dtype=args.f)
	width = args.W
	height = args.H
	chan = args.C
	cvt_method = args.c

	if (args.c == 1):
		im_data = im_data.reshape(height, width, chan)  ## interleave to planar
		im_out = np.zeros((chan, height, width), dtype=args.f)
		print("[interleave to planar] Input shape:", im_data.shape, "Output shape:", im_out.shape)

		for i in range(chan):
			im_out[i,:,:] = im_data[:,:,i]
	else:
		im_data = im_data.reshape(chan, height, width)  ## planar to interleave
		im_out = np.zeros((height, width, chan), dtype=args.f)
		print("[planar to interleave] Input shape:", im_data.shape, "Output shape:", im_out.shape)
		#im_out[:,:,0] = im_data[2,:,:]
		#im_out[:,:,1] = im_data[1,:,:]
		#im_out[:,:,2] = im_data[0,:,:]
		for i in range(chan):
			im_out[:,:,i] = im_data[i,:,:]

	im_out.tofile(args.o)

def cvt_planar_itl_by_2dim(args):
	im_data = np.fromfile(args.i, dtype=args.f)
	other_dim = 1
	chan = args.C
	cvt_method = args.c

	if (args.c == 1):
		im_data = im_data.reshape(-1, chan)  ## interleave to planar
		other_dim = im_data.shape[0]
		im_out = np.zeros((chan, other_dim), dtype=args.f)
		print("[interleave to planar] Input shape:", im_data.shape, "Output shape:", im_out.shape)

		for i in range(chan):
			im_out[i,:] = im_data[:,i]
	else:
		im_data = im_data.reshape(chan, -1)  ## planar to interleave
		other_dim = im_data.shape[1]
		im_out = np.zeros((other_dim, chan), dtype=args.f)
		print("[planar to interleave] Input shape:", im_data.shape, "Output shape:", im_out.shape)

		for i in range(chan):
			im_out[:,i] = im_data[i,:]

	im_out.tofile(args.o)

def init_param(args):
	parser = argparse.ArgumentParser(description="Convert channel between planar and interleave")
	parser.add_argument("-i", type=str, required=True, default="img.bin",
		help="input image filename")
	parser.add_argument("-f", type=str, required=False, default="uint8",
		help="input binary format: " + dtype_str)
	#parser.add_argument("-W", type=int, required=False,
	#	help="image width")
	#parser.add_argument("-H", type=int, required=False,
	#	help="image height")
	parser.add_argument("-C", type=int, required=True,
		help="image channel number, range [2|3]")
	parser.add_argument("-o", type=str, required=False,
		help="out bin filename")
	parser.add_argument("-c", type=int, required=False, default=0,
		help="color format convert method, 0: planar to interleave; 1: interleave to planar")
	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	cvt_planar_itl_by_2dim(args)
