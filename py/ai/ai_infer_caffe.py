#!/usr/bin/env python3

import os, sys, argparse
import caffe
import cv2
import numpy as np

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def inference_model(args):
	## Load model
	net = caffe.Net(args.p, args.m, caffe.TEST)

	## Load input
	input_shape = net.blobs[args.iname].data.shape
	print("Input name:", args.iname, ",shape: ", input_shape)
	input_data = np.fromfile(args.ibin, dtype=args.fmt)
	input_data = input_data.reshape(input_shape)
	net.blobs[args.iname].data[...] = input_data

	## inference model
	outs = net.forward()

	## save output
	print("output name: ", args.oname, ",shape: ", net.blobs[args.oname].data.shape)
	outs[args.oname].astype(np.float32).tofile("PC_{}.bin".format(args.oname))


def init_param(args):
	parser = argparse.ArgumentParser(description="Inference Caffe Model by feeding one input file")

	parser.add_argument("-p", type=str, required=True, default="sample.prototxt",
		help="caffe prototxt file")
	parser.add_argument("-m", type=str, required=True, default="sample.caffemodel",
		help="caffe weight file")

	parser.add_argument("-iname", type=str, required=True, default="data",
		help="input name")
	parser.add_argument("-ibin", type=str, required=True, default="input.bin",
		help="input binary file")
	parser.add_argument("-fmt", type=str, required=False, default="uint8",
		help="input binary format: " + dtype_str)

	parser.add_argument("-oname", type=str, required=True, default="out",
		help="output name")

	parser.add_argument("-v", action='store_true', required=False,
		help="Show verbose info")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	inference_model(args)
