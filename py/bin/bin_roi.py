#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import cv2

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def bin_roi(args):
	img = np.fromfile(args.b, dtype=args.f)
	width = args.W
	height = args.H

	img = img.reshape(-1, height, width)
	chan = img.shape[0]
	print("Input bin shape: ", img.shape)
	#print("Input bin shape[0,1,2]: {} {} {}".format(img.shape[0], img.shape[1], img.shape[2]))

	crop_arg = args.roi.split(',')
	xmin, ymin, cropW, cropH = list(map(int, crop_arg))
	xmax = xmin + cropW - 1
	ymax = ymin + cropH - 1
	print("ROI: (x, y, w, h) = ({:d}, {:d}, {:d}, {:d})".format(xmin, ymin, cropW, cropH))
	#print("Crop: (xmin, ymin, xmax, ymax) = ({:d}, {:d}, {:d}, {:d})".format(xmin, ymin, xmax, ymax))
	if ( ( ymax >= img.shape[1] ) or ( xmax >= img.shape[2] ) ):
		print('[Error] Crop size does not match the input image size. ""(Crop: [xmin, ymin, xmax, ymax]="\
		"[{:d},{:d},{:d},{:d}] / Image: [width, height]=[{:d},{:d}])'.format(xmin, ymin, xmax, ymax, img.shape[2], img.shape[1]))
		failQ.put(input_name)
		return

	img = img[:, ymin:ymax+1, xmin:xmax+1]  ## RGB Planar

	img.tofile(args.o)


def img_roi(args):
	img = cv2.imread(args.i)

	crop_arg = args.roi.split(',')
	xmin, ymin, cropW, cropH = list(map(int, crop_arg))
	xmax = xmin + cropW - 1
	ymax = ymin + cropH - 1
	#print("Input bin shape: ", img.shape)
	print("ROI: (x, y, w, h) = ({:d}, {:d}, {:d}, {:d})".format(xmin, ymin, cropW, cropH))
	#print("Crop: (xmin, ymin, xmax, ymax) = ({:d}, {:d}, {:d}, {:d})".format(xmin, ymin, xmax, ymax))
	if ( ( ymax >= img.shape[0] ) or ( xmax >= img.shape[1] ) ):
		print('[Error] Crop size does not match the input image size. ""(Crop: [xmin, ymin, xmax, ymax]="\
		"[{:d},{:d},{:d},{:d}] / Image: [width, height]=[{:d},{:d}])'.format(xmin, ymin, xmax, ymax, img.shape[1], img.shape[0]))
		failQ.put(input_name)
		return

	img = img[ymin:ymax+1, xmin:xmax+1, :]  ## RGB interlave

	cv2.imwrite(args.o, img)


def init_param(args):
	parser = argparse.ArgumentParser(description="Crop ROI on input image or binary")
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

	parser.add_argument("-roi", type=str, required=True,
		help="Crop ROI, format(x, y, w, h)")

	parser.add_argument("-o", type=str, required=True,
		help="out croped file")
	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])

	if (args.b is not None):
		bin_roi(args)
	else:
		img_roi(args)
