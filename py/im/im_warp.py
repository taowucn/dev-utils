#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import cv2

def do_warp(im, width, height, is_warpPersp):
	if (is_warpPersp):
		## Modify this points for warpPerspective
		points1 = np.float32([ [0,0], [width, 0], [0,height], [width, height] ])
		points2 = np.float32([ [0,0], [width * 0.7, height * 0.3], [width * 0.3, height * 0.7], [width * 0.6, height * 0.6] ])
		matix =  cv2.getPerspectiveTransform(points1, points2);
		print("warpPerspective Matrix:")
		print(matix.dtype)
		print(matix)
		_mat = matix.astype("float32")
		_mat.tofile("_warpPerspective.mat")

		return cv2.warpPerspective(im, matix, (width, height))
	else:
		## Modify this points for warpAffine
		points1 = np.float32([ [0,0], [width, 0], [0,height] ])
		points2 = np.float32([ [0,0], [width * 0.7, height * 0.3], [width * 0.3, height * 0.7] ])
		matix = cv2.getAffineTransform(points1, points2);
		#matix = cv2.invertAffineTransform(matix);

		print("warpAffine Matrix:")
		print(matix.dtype)
		print(matix)
		_mat = matix.astype("float32")
		_mat.tofile("_warpAffine.mat")

		return cv2.warpAffine(im, matix, (width, height),) # WARP_INVERSE_MAP

def warp_bin_file(args):
	im_data = np.fromfile(args.i, np.uint8)
	width = args.W
	height = args.H
	chan = args.C

	if ((width == 0) or (height == 0) or (chan == 0)):
		raise UserWarning("Invalid argument, %d, %d, %d" % (width, height, chan))

	im_data = im_data.reshape(chan, height, width)

	if (chan == 1):
		im = im_data.transpose(1,2,0)
	else:
		# rgb -> bgr;  (c, w, h) -> (h, w, c)
		im_data = im_data[::-1,:,:]
		im = im_data.transpose(1,2,0)

	im_warped = do_warp(im, width, height, args.p)

	if (chan == 1):
		im_warped.tofile(args.o)
	else:
		for i in range(3):
			im_data[i,:,:] = im_warped[:,:,2-i]
		im_data.tofile(args.o)

	if (args.d):
		cv2.namedWindow(args.o, 0)
		cv2.imshow(args.o, im_warped)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def warp_img_file(args):
	im = cv2.imread(args.i)
	height = im.shape[0]
	width = im.shape[1]

	im_warped = do_warp(im, width, height, args.p)

	cv2.imwrite(args.o, im_warped)

	if (args.d):
		cv2.namedWindow(args.o, 0)
		cv2.imshow(args.o, im_warped)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def im_warp(args):
	if (args.b):
		warp_bin_file(args)
	else:
		warp_img_file(args)

def init_param(args):
	parser = argparse.ArgumentParser(description="warpAffine or warpPerspestive. "
		"User need modify the Point2Point inside tool manually")
	parser.add_argument("-i", type=str, required=True, default="img.jpg",
		help="input image filename")
	parser.add_argument("-o", type=str, required=True,
		help="out image filename")

	## for choice warpAffine or warpPerspestive
	parser.add_argument("-p", action='store_true',
		help="warpAffine or warpPerspestive, default is warpAffine")

	## for choice warpAffine or warpPerspestive
	parser.add_argument("-d", action='store_true',
		help="Display image on window, imshow")

	## For binary rotate
	parser.add_argument("-b", action='store_true',
		help="input is image or binary format, default is image")
	parser.add_argument("-W", type=int, required=False,
		help="image width")
	parser.add_argument("-H", type=int, required=False,
		help="image height")
	parser.add_argument("-C", type=int, required=False,
		help="image channel number, range [1|3]")
	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	im_warp(args)
