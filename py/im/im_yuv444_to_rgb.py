#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import cv2
import math

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

BIAS_YUV2RGB = np.array([0, -128, -128], dtype=np.float32)
MATRIX_YUV2RGB = np.array([
                            [1, 0,        1.40625],
                            [1, -0.34375, -0.71875],
                            [1, 1.765625, 0]], dtype=np.float32)

BIAS_YUV2RGB_OPENCV = np.array([0, -128, -128], dtype=np.float32)
MATRIX_YUV2RGB_OPENCV = np.array([
                            [1, 0,        1.140015],
                            [1, -0.395020, -0.580094],
                            [1, 2.031982, 0]], dtype=np.float32)

def yuv444_to_rgb_planar(yuv, dc, dh, dw):
	#y = yuv[:dw*dh].reshape(dh, dw).astype(np.float32)
	#uv = yuv[dw*dh:].reshape(math.ceil(dh/2),math.ceil(dw/2),2)
	#u = cv2.resize(uv[:,:,0], (dw,dh), interpolation=cv2.INTER_NEAREST).astype(np.float32)
	#v = cv2.resize(uv[:,:,1], (dw,dh), interpolation=cv2.INTER_NEAREST).astype(np.float32)

	yuv444=yuv.reshape(dc, dh, dw).astype(np.float32)
	y = yuv444[0,:,:]
	u = yuv444[1,:,:]
	v = yuv444[2,:,:]

	#uv44=np.zeros((2, dh, dw), np.float32)
	#uv44[0,:,:]=u[:,:]
	#uv44[1,:,:]=v[:,:]

	#yuv444=np.concatenate(y.flatten(), u.flatten(), v.flatten())
	#yuv444.tofile("out_pc_{}x{}_fp32.yuv444".format(dw, dh))
	#yuv444.astype(np.uint8).tofile("out_pc_{}x{}.yuv444".format(dw, dh))
	#uv44.astype(np.uint8).tofile("out_pc_{}x{}.uv44".format(dw, dh))

	if (args.m == 0):
		rgb = np.zeros((dc, dh, dw), np.float32)
		#rgb[0,:,:] = 1*(y-0) + 1.40625*(v-128)
		#rgb[1,:,:] = 1*(y-0) - 0.34375*(u-128) - 0.71875*(v-128)
		#rgb[2,:,:] = 1*(y-0) + 1.765625*(u-128)
		rgb[0,:,:] = 1*(y-0) + 1.40625*(v-128)
		rgb[1,:,:] = 1*(y-0) - 0.34375*(u-128) - 0.71875*(v-128)
		rgb[2,:,:] = 1*(y-0) + 1.765625*(u-128)
	elif (args.m == 1):
		## the same result as  0
		#yuv444[0,:,:] += BIAS_YUV2RGB[0]
		#yuv444[1,:,:] += BIAS_YUV2RGB[1]
		#yuv444[2,:,:] += BIAS_YUV2RGB[2]
		#yuv444 = np.add(yuv444, BIAS_YUV2RGB)
		yuv444_itl = yuv444.transpose(1, 2, 0)
		yuv444_itl += BIAS_YUV2RGB
		#yuv444_itl -= [0, 128, 128]
		#print(yuv444.shape, yuv444_itl.shape)
		rgb_itl = np.matmul(yuv444_itl.copy(), MATRIX_YUV2RGB.T) ## dot same as matmul, need tranpose.matrix
		rgb = rgb_itl.transpose(2, 0, 1)
	elif (args.m == 2):
		rgb = np.zeros((dc, dh, dw), np.float32)
		rgb[0,:,:] = 1.164*(y-16) + 1.596*(v-128)
		rgb[1,:,:] = 1.164*(y-16) - 0.391*(u-128) - 0.813*(v-128)
		rgb[2,:,:] = 1.164*(y-16) + 2.018*(u-128)
	elif (args.m == 3):
		## the same result as 2
		yuv444_itl = yuv444.transpose(1, 2, 0)
		yuv444_itl += BIAS_YUV2RGB_OPENCV
		rgb_itl = np.matmul(yuv444_itl.copy(), MATRIX_YUV2RGB_OPENCV.T) ## dot same as matmul, need tranpose.matrix
		rgb = rgb_itl.transpose(2, 0, 1)
	else:
		raise UserWarning("Unsupport method, support [0,1,2,3]" % (args.m))

	#rgb.tofile("a_fp32.bin")
	#rgb.tofile("out_pc_{}x{}x{}_fp32_before_clip.rgb".format(dc, dw, dh))
	rgb = rgb.clip(0.0, 255.0)
	#rgb.tofile("b_fp32.bin")

	#return rgb.reshape(-1).astype(np.float32)
	return rgb.astype(np.uint8)

def yuv_to_rgb(args):
	yuv_data = np.fromfile(args.i, np.uint8)
	width = args.W
	height = args.H
	chan = 3
	rgb_color_fmt = args.c
	yuv_color_fmt = args.y

	## src
	if (yuv_color_fmt == 0):
		rgb_data_planar = yuv444_to_rgb_planar(yuv_data, chan, height, width)
	else:
		raise UserWarning("Unsupport YUV color format : %d, only YUV444 [0]" % (yuv_color_fmt))

	## dst
	if (rgb_color_fmt == 0):  ## RGB planar
		rgb_data = rgb_data_planar
	else:
		raise UserWarning("Unsupport RGB color format : %d, only RGB planar [2]" % (yuv_color_fmt))

	rgb_data.tofile(args.o)

def init_param(args):
	parser = argparse.ArgumentParser(description="Transfer YUV444 to RGB, only support uint8 data format")
	parser.add_argument("-i", type=str, required=True, default="img.bin",
		help="input NV12 binary (3, H, W) filename, only uint8 format")
	parser.add_argument("-W", type=int, required=True,
		help="image width")
	parser.add_argument("-H", type=int, required=True,
		help="image height")
	parser.add_argument("-o", type=str, required=False,
		help="out YUV filename")
	parser.add_argument("-c", type=int, required=False, default=0,
		help="color format of RGB, 0: RGB on Channel (default); 1: RGB is interleave; 2: BGR.")
	parser.add_argument("-y", type=int, required=False, default=2,
		help="color format of YUV, 0: YUV444;  1: IYUV (YUV420 planar);  2: NV12 (YUV420 semi-planar).")
	parser.add_argument("-m", type=int, required=False, default=3,
		help="The cvtcolor method (default 1). 0/1: bias[0,-128,-128], matrix;  2/3: bias[-16, -128, -128], matrix in opencv")
	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	yuv_to_rgb(args)

