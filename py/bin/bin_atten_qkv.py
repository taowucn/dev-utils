#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8|bool>"

def bin_atten_qkv(args):
	import torch
	import math

	data_src_q = np.fromfile(args.iq, dtype=args.f)
	data_src_k = np.fromfile(args.ik, dtype=args.f)
	data_src_v = np.fromfile(args.iv, dtype=args.f)

	data_src_q = data_src_q.astype(np.float32)
	data_src_k = data_src_k.astype(np.float32)
	data_src_v = data_src_v.astype(np.float32)

	if (args.q):
		data_src_q = data_src_q * pow(2, -(args.q))
		data_src_k = data_src_k * pow(2, -(args.q))
		data_src_v = data_src_v * pow(2, -(args.q))

	# model config
	m_config = args.s.split(',')
	cfg_seq_len, cfg_n_heads, cfg_head_dim = list(map(int, m_config))  # (seq_len, n_heads, head_dim)
	print("Model config: (seq_len, n_heads, head_dim) = ({:d}, {:d}, {:d})".format(cfg_seq_len, cfg_n_heads, cfg_head_dim))

	data_src_q = data_src_q.reshape(-1, cfg_n_heads, 1, cfg_head_dim)
	data_src_k = data_src_k.reshape(-1, cfg_seq_len, cfg_n_heads, cfg_head_dim)
	data_src_v = data_src_v.reshape(-1, cfg_seq_len, cfg_n_heads, cfg_head_dim)
	print("Query shape:", data_src_q.shape)
	print("Key shape:", data_src_k.shape)
	print("Value shape:", data_src_v.shape)

	# to pytorch tensor
	xq = torch.from_numpy(data_src_q)
	xk = torch.from_numpy(data_src_k)
	xv = torch.from_numpy(data_src_v)

	# transpose
	#if (args.t):
	#	print("input transpose")

       # computation
	xk = torch.permute(xk, (0, 2, 3, 1))  # (1, 256, 6, 48) -> (1, 6, 48, 256)
	xv = torch.permute(xv, (0, 2, 1, 3))  # (1, 256, 6, 48) -> (1, 6, 256, 48)
	print("Transposed Key   shape:", xk.shape)
	print("Transposed Value shape:", xv.shape)
	# matmul
	scores = torch.matmul(xq, xk) / math.sqrt(cfg_head_dim)
	print("scores shape:", scores.shape)
	scores = torch.nn.functional.softmax(scores, dim=-1).type_as(xq)
	print("scores shape after softmax:", scores.shape)
	output = torch.matmul(scores, xv)

	if (args.q):
		output = output * pow(2, (args.q))

	data_dst = output.numpy()
	data_dst = data_dst.astype(args.f)
	data_dst.tofile(args.o)

	print('Ouput shape: ', data_dst.shape)
	print('Attenion QKV File: {} {} {} = {}. Format: {}, Q: {}.'.format(args.iq, args.ik, args.iv, args.o, args.f, args.q))


def init_param(args):
	parser = argparse.ArgumentParser(description="Compute Transformer Attention QKV dot product result for binary file")

	parser.add_argument("-iq", type=str, required=True, default="query.bin",
		help="query filename, shape: (1, n_heads, 1, head_dim) e.g. Llama15M (1, 6, 1, 48)")
	parser.add_argument("-ik", type=str, required=True, default="key_cache.bin",
		help="key cache filename, shape: (1, seq_len, n_heads, head_dim) e.g. Llama15M (1, 256, 6, 48)")
	parser.add_argument("-iv", type=str, required=True, default="value_cache.bin",
		help="value cache filename, shape: (1, seq_len, n_heads, head_dim) e.g. Llama15M (1, 256, 6, 48)")

	parser.add_argument("-s", type=str, required=True,
		help="model config. (seq_len, n_heads, head_dim), e.g. Llama15M (256, 6, 48) ")

	#parser.add_argument("-t", action='store_true', required=False,
	#	help="input 2 if need tranpose")

	parser.add_argument("-o", type=str, required=True, default="output.bin",
		help="ouput binary filename. softmax(Q @ K.T/sqrt(head_size)) @ V = output")

	parser.add_argument("-f", type=str, required=True, default="float32",
		help="input and output binary format: " + dtype_str)

	parser.add_argument("-q", type=int, required=False, default=0,
		help="input and output binary Q (expoffset) value ")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_atten_qkv(args)
