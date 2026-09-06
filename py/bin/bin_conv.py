#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8|bool> "

dtype_ext_str = "<bf16|fp8_e4m3|fp8_e5m2> "
EXT_DTYPES = {"bf16", "fp8_e4m3", "fp8_e5m2"}
dconv_fpx = None

# load data in any format and convert it to float32
def load_as_float32(filepath, dtype):
	if dtype == "bf16":
		raw = np.fromfile(filepath, dtype=np.uint16)
		return np.vectorize(dconv_fpx.bf16_to_float32, otypes=[np.float32])(raw)
	if dtype == "fp8_e4m3":
		raw = np.fromfile(filepath, dtype=np.uint8)
		return np.vectorize(dconv_fpx.fp8_e4m3_to_float32, otypes=[np.float32])(raw)
	if dtype == "fp8_e5m2":
		raw = np.fromfile(filepath, dtype=np.uint8)
		return np.vectorize(dconv_fpx.fp8_e5m2_to_float32, otypes=[np.float32])(raw)
	return np.fromfile(filepath, dtype=dtype).astype(np.float32)

# save data from float32 to any format
def save_from_float32(data, filepath, dtype):
	if dtype == "bf16":
		converted = np.vectorize(dconv_fpx.float32_to_bf16, otypes=[np.uint16])(data)
	elif dtype == "fp8_e4m3":
		converted = np.vectorize(dconv_fpx.float32_to_fp8_e4m3, otypes=[np.uint8])(data)
	elif dtype == "fp8_e5m2":
		converted = np.vectorize(dconv_fpx.float32_to_fp8_e5m2, otypes=[np.uint8])(data)
	else:
		converted = data.astype(dtype)
	converted.tofile(filepath)

def bin_conv(args):
	if args.f not in EXT_DTYPES and args.f not in np.sctypeDict:
		raise ValueError("Unsupported source format: {}".format(args.f))
	if args.t not in EXT_DTYPES and args.t not in np.sctypeDict:
		raise ValueError("Unsupported destination format: {}".format(args.t))
	if args.f in EXT_DTYPES or args.t in EXT_DTYPES:
		print("import data_conv_fp_ext")
		global dconv_fpx
		import data_conv_fp_ext as dconv_fpx

	data_src = load_as_float32(args.i, args.f)
	if (args.q):
		data_src = data_src * pow(2, args.q)

	save_from_float32(data_src, args.o, args.t)
	print('Convert File: {} -> {}, Format: {} -> {}(Q:{})'.format(args.i, args.o, args.f, args.t, args.q))

def init_param(args):
	parser = argparse.ArgumentParser(description="Convert binary file from specific format to another format "
									"(1.0.1), support bf16, fp8")
	parser.add_argument("-i", type=str, required=True, default="input.bin",
		help="input image filename")
	parser.add_argument("-o", type=str, required=True, default="output.bin",
		help="ouput binary filename")
	parser.add_argument("-f", type=str, required=True, default="fp32",
		help="src binary format: " + dtype_str + dtype_ext_str)
	parser.add_argument("-t", type=str, required=True, default="fp32",
		help="dest binary format: " + dtype_str + dtype_ext_str)
	parser.add_argument("-q", type=int, required=False, default=0,
		help="dest binary Q value")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_conv(args)
