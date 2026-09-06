#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8|bool>"

def bin_act_silu(args):
	import torch
	#from torch import nn

	data_src = np.fromfile(args.i, dtype=args.f)
	data_src = data_src.astype(np.float32)
	if (args.q):
		data_src = data_src * pow(2, -(args.q))

	## pytorch silu
	src_tensor = torch.from_numpy(data_src)
	dst_tensor = torch.nn.functional.silu(src_tensor)

	data_dst = dst_tensor.numpy()
	if (args.q):
		data_dst = data_dst * pow(2, (args.q))

	data_dst = data_dst.astype(args.f)
	data_dst.tofile(args.o)

	print('SiLU File: {} {} Format: {} Q: {} shape:{} '.format(args.i, args.o, args.f, args.q, data_dst.shape))

def init_param(args):
	parser = argparse.ArgumentParser(description="Compute SiLU act function for binary file")
	parser.add_argument("-i", type=str, required=True, default="input.bin",
		help="input image filename")
	parser.add_argument("-o", type=str, required=True, default="output.bin",
		help="ouput binary filename")

	parser.add_argument("-f", type=str, required=True, default="float32",
		help="input and output binary format: " + dtype_str)
	parser.add_argument("-q", type=int, required=False, default=0,
		help="input and output binary Q (expoffset) value ")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_act_silu(args)
