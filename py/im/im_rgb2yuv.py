#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import cv2

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def rgb_to_yuv(args):
	im_data = np.fromfile(args.i, dtype="uint8")
	width = args.W
	height = args.H
	chan = 3
	rgb_color_fmt = args.c
	yuv_color_fmt = args.y

	if (rgb_color_fmt == 1): # RGB interleave
		im_data = im_data.reshape(height, width, chan)
	else: # RGB / BGR planer
		im_data = im_data.reshape(chan, height, width)

	print("Input shape: ", im_data.shape)
	im_rgb_itl = np.zeros((height, width, chan), dtype="uint8")

	if (rgb_color_fmt == 1):
		im_rgb_itl = im_data
	elif (rgb_color_fmt == 0):
		im_rgb_itl[:,:,0] = im_data[0,:,:]
		im_rgb_itl[:,:,1] = im_data[1,:,:]
		im_rgb_itl[:,:,2] = im_data[2,:,:]
	elif (rgb_color_fmt == 2):
		im_rgb_itl[:,:,0] = im_data[2,:,:]
		im_rgb_itl[:,:,1] = im_data[1,:,:]
		im_rgb_itl[:,:,2] = im_data[0,:,:]
	else:
		raise UserWarning("Unsupport color format : %d, range [0,1,2]" % (rgb_color_fmt))

	if (yuv_color_fmt == 0):
		yuv_data = cv2.cvtColor(im_rgb_itl, cv2.COLOR_RGB2YUV) ## YUV444
		yuv_data.tofile(args.o)
	elif (yuv_color_fmt == 1):
		yuv_data = cv2.cvtColor(im_rgb_itl, cv2.COLOR_RGB2YUV_I420) ## IYUV, same as COLOR_RGB2YUV_IYUV
		yuv_data.tofile(args.o)
	elif (yuv_color_fmt == 2):
		yuv_data = cv2.cvtColor(im_rgb_itl, cv2.COLOR_RGB2YUV_I420) ## IYUV

		## Split Y and UV
		y_data = yuv_data.reshape(-1)[:height * width]
		uv_data = yuv_data.reshape(-1)[height * width:]

		uv_data = uv_data.reshape(2, -1)
		other_dim = uv_data.shape[1]
		uv_itl = np.zeros((other_dim, 2), np.uint8)
		#print("UV other_dim:", other_dim)

		## UV planar to interleave
		uv_itl[:,0] = uv_data[0,:]
		uv_itl[:,1] = uv_data[1,:]

		## concat Y an UV
		yuv_data = np.concatenate((y_data.flatten(), uv_itl.flatten()))

		yuv_data.tofile(args.o)

def init_param(args):
	parser = argparse.ArgumentParser(description="Transfer RGB binary to YUV (YUV444, NV12 ... ), only support uint8 data format")
	parser.add_argument("-i", type=str, required=True, default="img.bin",
		help="input RGB binary (3, H, W) filename, only uint8 format")
	parser.add_argument("-W", type=int, required=True,
		help="image width")
	parser.add_argument("-H", type=int, required=True,
		help="image height")
	parser.add_argument("-o", type=str, required=False,
		help="out YUV filename")
	parser.add_argument("-c", type=int, required=False, default=0,
		help="color format of RGB, 0: RGB on Channel (default); 1: RGB is interleave; 2: BGR.")
	parser.add_argument("-y", type=int, required=False, default=0,
		help="color format of YUV, 0: YUV444;  1: IYUV (YUV420 planar);  2: NV12 (YUV420 semi-planar).")
	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	rgb_to_yuv(args)
