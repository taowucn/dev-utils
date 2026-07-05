#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

from math import nan
from scipy import linalg
from scipy import spatial as sp

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

def check_fsize(f1, f2):
	if (len(f1) != len(f2)):
		print("Two file size: ", len(f1), len(f2))
		return False
	else:
		return True

def max_absolute_error(f1, f2):
	diff = np.abs(f1 - f2)
	max_abs_err = np.max(diff)
	#diff = diff.reshape(3,1080,1920)
	pos_abs_err = np.argmax(diff, axis=0)
	return (max_abs_err, pos_abs_err)

def relative_error(f1, f2):
	diff = f1 - f2
	#relative_err = 100*np.where(np.isclose(f1, 0, rtol=0.0001), f2, np.absolute(np.divide(diff, f1)))
	#relative_err = np.where(np.isclose(f1, 0, rtol=0.0001), f2, np.absolute(np.divide(diff, f1)))
	#is_close = np.isclose(f1, f2, rtol=1.e-3, atol=1.e-3, equal_nan=False)
	#all_close = np.allclose(f1, f2, rtol=1.e-3, atol=1.e-3, equal_nan=False)
	relative_err = diff
	return relative_err

def max_relative_error(f1, f2):
	relative_err = relative_error(f1, f2)
	max_relative_err = np.max(relative_err)
	pos_max_rtl_err = np.argmax(relative_err)
	#print(pos_max_rtl_err)
	return (max_relative_err, pos_max_rtl_err)

def avg_relative_error(f1, f2):
    rel_err = relative_error(f1, f2)
    avg_rel_err = np.average(rel_err)
    return avg_rel_err

def cosine_distance(f1, f2):
	f1_fp64 = f1.astype(np.float64, copy=False)
	f2_fp64 = f2.astype(np.float64, copy=False)
	eps = np.zeros_like(f1_fp64)

	if len(f1_fp64) == 1 and len(f2_fp64) == 1:
		return "#N/A (scalar tensors)"
	if (f1_fp64 == 0).all():
		eps = np.zeros_like(f1_fp64)
		eps[0] += np.finfo(f1_fp64.dtype).eps
	f1_fp64 = f1_fp64 + eps
	if (f2_fp64 == 0).all():
		eps = np.zeros_like(f2_fp64)
		eps[0] += np.finfo(f2_fp64.dtype).eps
	f2_fp64 = f2_fp64 + eps
	return sp.distance.cosine(f1_fp64, f2_fp64)

def psnr(f1, f2):
	f1 = f1.astype(np.float32, copy=False)
	f2 = f2.astype(np.float32, copy=False)
	mse = np.mean((f1 - f2) ** 2)
	max_value = max(np.max(np.abs(f1)), np.max(np.abs(f2)))
	if mse == 0.:
		return "+Inf (mse == 0.)"
	return 20 * np.log10(max_value / np.sqrt(mse))

def pearson_correlation(f1, f2):

    if np.isinf(f1).any() or np.isinf(f2).any():
        return "#N/A (tensors contain inf values)"

    if np.isnan(f1).any() or np.isnan(f2).any():
        return "#N/A (tensors contain nan values)"

    if len(f1) == 1 and len(f2) == 1:
        return "#N/A (scalar tensors)"

    # Undefined correlation coefficient for constant tensors
    if np.ptp(f1) == 0:
        return "#N/A (var(x) == 0 for pearsonr(x,y))"
    if np.ptp(f2) == 0:
        return "#N/A (var(y) == 0 for pearsonr(x,y))"

    # this expression ensures that the data type is at least 64 bit floating
    # point. It might have more precision if the input is, for example, np.longdouble.
    dtype = type(1.0 + f1[0] + f2[0])
    x = f1.astype(dtype, copy=False)
    y = f2.astype(dtype, copy=False)
    if len(x) == len(y) == 2:
        return dtype(np.sign(x[1] - x[0])*np.sign(y[1] - y[0]))

    if np.array_equal(x, y):
        # Workaround for unavoidable numerical imprecision in numpy dot product,
        # which uses the optimized coppersmith-winograd gemm algorithm.
        # See https://stackoverflow.com/questions/58740925/why-is-np-dot-imprecise-n-dim-arrays
        return 1.0

    xmean = x.mean(dtype=dtype)
    ymean = y.mean(dtype=dtype)
    xm = x.astype(dtype) - xmean
    ym = y.astype(dtype) - ymean

    # scipy.linalg.norm is better at preventing overflows
    normxm = linalg.norm(xm)
    normym = linalg.norm(ym)
    r = np.dot(xm/normxm, ym/normym)
    r = max(min(r, 1.0), -1.0)
    return r

def bin_err(args):
	data_a = np.fromfile(args.f1, dtype=args.t1)
	data_b = np.fromfile(args.f2, dtype=args.t2)

	if (data_a.dtype) == (data_b.dtype):
		print("Two files use the same dtype: ", args.t1, args.t2)
		if not check_fsize(data_a, data_b):
			print("Two files size different, error")
			return
	else:
		print("Two files use different dtype: ", args.t1, args.t2)

	data_fa = data_a.astype(np.float32)
	data_fb = data_b.astype(np.float32)

	if (args.q1):
		data_fa = data_fa/pow(2, args.q1)
	if (args.q2):
		data_fb = data_fb/pow(2, args.q2)

	if (args.v):
		print("--------------", args.f1)
		print(data_fa)
		print("--------------", args.f2)
		print(data_fb)
		print("============")

	max_abs_err, pos_abs_err = max_absolute_error(data_fa, data_fb)
	#relative_err = relative_error(data_fa, data_fb)
	max_relative_err, pos_max_rtl_err = max_relative_error(data_fa, data_fb)
	avg_relative_err = avg_relative_error(data_fa, data_fb)
	cosine_distance_err = cosine_distance(data_fa, data_fb)
	pearson_correlation_err = pearson_correlation(data_fa, data_fb)
	psnr_err = psnr(data_fa, data_fb)

	## print result
	print("max_abs_err:", max_abs_err, ", at position: ", pos_abs_err, ", left - right:", data_fa[pos_abs_err], "-", data_fb[pos_abs_err])
	#print("pos_abs_err:", pos_abs_err)
	#print("relative_err:", relative_err)
	print("max_rel_err:", max_relative_err, ", at postion: ", pos_max_rtl_err, ", left - right:", data_fa[pos_max_rtl_err], "-", data_fb[pos_max_rtl_err])
	print("avg_rel_err:", avg_relative_err)
	print("cosine_dist:", cosine_distance_err)
	print("pearson:", pearson_correlation_err)
	print("psnr:", psnr_err)

def init_param(args):
	parser = argparse.ArgumentParser(description="Compare two binary file error with corrsponding format")
	parser.add_argument("-f1", type=str, required=True, default="a.bin",
		help="input binary 1 filename")
	parser.add_argument("-f2", type=str, required=True, default="b.bin",
		help="input binary 2 filename")
	parser.add_argument("-t1", type=str, required=True, default="fp32",
		help="input binary 1 format: " + dtype_str)
	parser.add_argument("-t2", type=str, required=True, default="fp32",
		help="input binary 2 format: " + dtype_str)
	parser.add_argument("-q1", type=int, required=False,
		help="Q value for quantized data 1")
	parser.add_argument("-q2", type=int, required=False,
		help="Q value for quantized data 2")
	parser.add_argument("-v", action='store_true', required=False,
		help="Show data in float32")
	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_err(args)
