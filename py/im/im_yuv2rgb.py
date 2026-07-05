#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import cv2

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def yuv_to_rgb(args):
	yuv_data = np.fromfile(args.i, np.uint8) #.astype(np.float32)
	width = args.W
	height = args.H
	chan = 3
	rgb_color_fmt = args.c
	yuv_color_fmt = args.y

	## src
	if (yuv_color_fmt == 0):
		yuv_data = yuv_data.reshape(3, height, width)

		# YCbCr formlua; Cb = U; Cr = V;
		#yuv_data_yvu = np.zeros((chan, height, width), dtype=np.uint8)
		#yuv_data_yvu[0,:,:] = yuv_data[0,:,:]
		#yuv_data_yvu[1,:,:] = yuv_data[2,:,:]
		#yuv_data_yvu[2,:,:] = yuv_data[1,:,:]
		#print(yuv_data_yvu.shape)

		yuv_data_itl = yuv_data.transpose(1, 2, 0)
		rgb_data_itl = cv2.cvtColor(yuv_data_itl, cv2.COLOR_YUV2RGB) ## YUV444 (yyyy uuuu vvvv)
	elif (yuv_color_fmt == 1):
		yuv_data_itl = yuv_data.reshape(int(height * 1.5), width)
		rgb_data_itl = cv2.cvtColor(yuv_data_itl, cv2.COLOR_YUV2RGB_IYUV) ## IYUV
	elif (yuv_color_fmt == 2):
		#y_arr = yuv_data[:height*width]
		#uv_arr = yuv_data[height*width:]
		#rgb_data_itl = cv2.cvtColorTwoPlane(y_arr, uv_arr, cv2.COLOR_YUV2RGB_NV12) #.astype(np.float32)
		yuv_data_itl = yuv_data.reshape(int(height * 1.5), width)
		rgb_data_itl = cv2.cvtColor(yuv_data_itl, cv2.COLOR_YUV2RGB_NV12) ## NV12
	else:
		raise UserWarning("Unsupport YUV color format : %d, range [0,1,2]" % (yuv_color_fmt))

	## dst
	rgb_data = np.zeros((chan, height, width), dtype=np.float32)
	if (rgb_color_fmt == 0):  ## RGB planar
		rgb_data[0,:,:] = rgb_data_itl[:,:,0]
		rgb_data[1,:,:] = rgb_data_itl[:,:,1]
		rgb_data[2,:,:] = rgb_data_itl[:,:,2]
	elif (rgb_color_fmt == 1):  ## RGB interleave
		rgb_data = rgb_data_itl
	elif (rgb_color_fmt == 2):  ## BGR planar
		rgb_data[2,:,:] = rgb_data_itl[:,:,0]
		rgb_data[1,:,:] = rgb_data_itl[:,:,1]
		rgb_data[0,:,:] = rgb_data_itl[:,:,2]

	fname = args.o + ".fp32"
	rgb_data.tofile(fname)
	rgb_data.astype(np.uint8).tofile(args.o)

def init_param(args):
	parser = argparse.ArgumentParser(description="Transfer YUV (YUV444, NV12 ... ) to RGB by OpenCV, only support uint8 data format")
	parser.add_argument("-i", type=str, required=True, default="img.bin",
		help="input YUV binary (3, H, W) filename, only uint8 format")
	parser.add_argument("-W", type=int, required=True,
		help="image width")
	parser.add_argument("-H", type=int, required=True,
		help="image height")
	parser.add_argument("-o", type=str, required=False,
		help="out YUV filename, only uint8 format")
	parser.add_argument("-c", type=int, required=False, default=0,
		help="color format of RGB, 0: RGB on Channel (default); 1: RGB is interleave; 2: BGR.")
	parser.add_argument("-y", type=int, required=False, default=0,
		help="color format of YUV, 0: YUV444;  1: IYUV (YUV420 planar);  2: NV12 (YUV420 semi-planar).")
	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	yuv_to_rgb(args)
