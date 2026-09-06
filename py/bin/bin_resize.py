#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import cv2

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def bin_resize(args):
	img = np.fromfile(args.b, dtype=args.f)

	input_dim_arg = args.src.split(',')
	input_W, input_H = list(map(int, input_dim_arg))
	
	output_dim_arg = args.dst.split(',')
	output_W, output_H = list(map(int, output_dim_arg))
	
	img = img.reshape(-1, input_H, input_W)
	print("Input bin shape: ", img.shape)
	print("Output bin shape (H, W): ", output_H, output_W)
	
	img_itl = img.transpose(1,2,0)
	out_img_itl = cv2.resize(img_itl, (output_W, output_H))
	out_img = out_img_itl.transpose(2,0,1)
	out_img.tofile(args.o)


def img_resize(args):
	img = cv2.imread(args.i)
	
	output_dim_arg = args.src.split(',')
	output_W, output_H = list(map(int, output_dim_arg))
	print("Output bin shape (H, W): ", output_H, output_W)
	
	out_img = cv2.resize(img, (output_W, output_H))
	cv2.imwrite(args.o, img)


def init_param(args):
	parser = argparse.ArgumentParser(description="Resize on input image or binary, RGB is on planer")
	parser.add_argument("-i", type=str, required=False,
		help="input image(*.jpg, *.png) filename")

	parser.add_argument("-b", type=str, required=False,
		help="input RGB Planar binary (1|3, H, W) filename")
	parser.add_argument("-f", type=str, required=False, default="uint8",
		help="input binary format: " + dtype_str)

	parser.add_argument("-src", type=str, required=False,
		help="Input resolution, format(w, h), need specify this when use binary")

	parser.add_argument("-dst", type=str, required=True,
		help="Output resolution, format(w, h), e.g:  1920,1080")

	parser.add_argument("-o", type=str, required=True,
		help="out resize file")
	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])

	if (args.b is not None):
		bin_resize(args)
	else:
		img_resize(args)
