#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import cv2

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def bin_resize(args):
	img = np.fromfile(args.b, dtype=args.f)
	width = args.W
	height = args.H

	img = img.reshape(-1, height, width)
	chan = img.shape[0]
	print("Input bin shape: ", img.shape)

	resize_arg = args.resized.split(',')
	cropW, cropH = list(map(int, resize_arg))
	print("Resized: (w, h) = ({:d}, {:d})".format(cropW, cropH))

	if (chan == 1):
		img = img.reshape(height, width)
		out_img = cv2.resize(img, (cropW, cropH))
	elif (chan == 3):
		r = img[0,:,:]
		g = img[1,:,:]
		b = img[2,:,:]
		out_r = cv2.resize(r, (cropW, cropH))
		out_g = cv2.resize(g, (cropW, cropH))
		out_b = cv2.resize(b, (cropW, cropH))

		out_img = np.zeros((3, cropH, cropW), dtype=args.f)
		out_img[0,:,:] = out_r
		out_img[1,:,:] = out_g
		out_img[2,:,:] = out_b
	else:
		raise UserWarning("Unsupport channel: %d, range [1|3]" % (chan))
		return;

	out_img.tofile(args.o)

def img_resize(args):
	img = cv2.imread(args.i)

	resize_arg = args.resized.split(',')
	cropW, cropH = list(map(int, resize_arg))
	#print("Input bin shape: ", img.shape)
	print("Resized: (w, h) = ({:d}, {:d})".format(cropW, cropH))

	out_img = cv2.resize(img, (cropW, cropH))
	cv2.imwrite(args.o, out_img)


def init_param(args):
	parser = argparse.ArgumentParser(description="Resize input image or binary")
	parser.add_argument("-i", type=str, required=False,
		help="input image(*.jpg, *.png) filename")

	parser.add_argument("-b", type=str, required=False,
		help="input RGB Planar binary (1|3, H, W) filename")
	parser.add_argument("-f", type=str, required=False, default="uint8",
		help="input binary format: " + dtype_str)

	parser.add_argument("-W", type=int, required=False,
		help="image width, need specify this when use binary input")
	parser.add_argument("-H", type=int, required=False,
		help="image height, need specify this when use binary input")

	parser.add_argument("-resized", type=str, required=True,
		help="Resized shape, format(w, h)")

	parser.add_argument("-o", type=str, required=True,
		help="out resized file")
	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])

	if (args.b is not None):
		bin_resize(args)
	else:
		img_resize(args)
