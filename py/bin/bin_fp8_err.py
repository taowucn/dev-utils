#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

from math import nan, inf
from scipy import linalg
from scipy import spatial as sp

dtype_str = "<e4m3|e5m2|float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8> "

dtype_ext_str = "<bf16|fp8_e4m3|fp8_e5m2> "

# fp8 format: (exp_bits, mantissa_bits, exp_bias, has_inf), bit layout [sign|exp|mantissa]
#   e4m3: OCP FP8 E4M3, no infinity, NaN is S.1111.111, max normal 448
#   e5m2: OCP FP8 E5M2, IEEE-754 like, Inf is S.11111.00, max normal 57344
FP8_FORMATS = {
	"e4m3": (4, 3,  7, False),
	"e5m2": (5, 2, 15, True),
}

FP8_ALIAS = {
	"fp8":           "e4m3",
	"fp8_e4m3":      "e4m3",
	"fp8_e5m2":      "e5m2",
}

def norm_dtype(t):
	tl = t.lower()
	return FP8_ALIAS.get(tl, tl)

def fp8_value_table(fmt):
	"""Decode all 256 fp8 codes of one format into float64."""
	ebits, mbits, bias, has_inf = FP8_FORMATS[fmt]
	e_all = (1 << ebits) - 1
	m_all = (1 << mbits) - 1
	tbl = np.zeros(256, dtype=np.float64)
	for code in range(256):
		sign = -1.0 if (code >> 7) & 1 else 1.0
		exp = (code >> mbits) & e_all
		mant = code & m_all
		if exp == e_all and (has_inf or mant == m_all):
			tbl[code] = sign * inf if (has_inf and mant == 0) else nan
			continue
		if exp == 0:
			val = (mant / (1 << mbits)) * 2.0 ** (1 - bias)
		else:
			val = (1.0 + mant / (1 << mbits)) * 2.0 ** (exp - bias)
		tbl[code] = sign * val
	return tbl

RANK_INVALID = -10000

def fp8_rank_table(fmt):
	"""Map each finite fp8 code to a monotonic rank, so |rank(a)-rank(b)| is the ULP distance.

	fp8 is sign-magnitude, so magnitude codes 0..127 are already value-ordered; the negative
	half mirrors them. +0 and -0 share rank 0, which is what a ULP distance should report.
	"""
	tbl = fp8_value_table(fmt)
	rank = np.full(256, RANK_INVALID, dtype=np.int32)
	for code in range(256):
		if not np.isfinite(tbl[code]):
			continue
		mag = code & 0x7F
		rank[code] = -mag if (code >> 7) & 1 else mag
	return rank

def fp8_decode(raw, fmt):
	return fp8_value_table(fmt)[raw]

def fp8_encode(data, fmt):
	"""Round float data to the nearest representable fp8 code (ties to even code)."""
	tbl = fp8_value_table(fmt)
	finite = np.nonzero(np.isfinite(tbl))[0]
	vals = tbl[finite]
	order = np.argsort(vals, kind="stable")
	vals = vals[order]
	codes = finite[order]
	x = np.asarray(data, dtype=np.float64).ravel()
	idx = np.searchsorted(vals, x)
	idx = np.clip(idx, 1, len(vals) - 1)
	lo, hi = vals[idx - 1], vals[idx]
	pick = np.where(np.abs(x - lo) <= np.abs(hi - x), idx - 1, idx)
	out = codes[pick].astype(np.uint8)
	out[np.isnan(x)] = 0x7F if fmt == "e4m3" else 0x7E
	return out

def load_bin(fname, dtype):
	"""Return (float64 view, raw uint8 codes or None, count)."""
	dt = norm_dtype(dtype)
	if dt in FP8_FORMATS:
		raw = np.fromfile(fname, dtype=np.uint8)
		return fp8_decode(raw, dt).astype(np.float64), raw, dt
	data = np.fromfile(fname, dtype=dt)
	return data.astype(np.float64), None, dt

def check_fsize(f1, f2):
	if (len(f1) != len(f2)):
		print("Two file size: ", len(f1), len(f2))
		return False
	else:
		return True

def max_absolute_error(f1, f2):
	diff = np.abs(f1 - f2)
	max_abs_err = np.max(diff)
	pos_abs_err = np.argmax(diff)
	return (max_abs_err, pos_abs_err)

def relative_error(f1, f2):
	"""|a-b| / |a|, falling back to |a-b| where the reference is 0."""
	diff = np.abs(f1 - f2)
	denom = np.abs(f1)
	return np.where(denom > 0, diff / np.where(denom > 0, denom, 1.0), diff)

def max_relative_error(f1, f2):
	rel_err = relative_error(f1, f2)
	pos = np.argmax(rel_err)
	return (rel_err[pos], pos)

def avg_relative_error(f1, f2):
	return np.average(relative_error(f1, f2))

def cosine_distance(f1, f2):
	f1_fp64 = f1.astype(np.float64, copy=False)
	f2_fp64 = f2.astype(np.float64, copy=False)

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
	mse = np.mean((f1 - f2) ** 2)
	max_value = max(np.max(np.abs(f1)), np.max(np.abs(f2)))
	if mse == 0.:
		return "+Inf (mse == 0.)"
	if max_value == 0.:
		return "#N/A (all zero tensors)"
	return 20 * np.log10(max_value / np.sqrt(mse))

def pearson_correlation(f1, f2):
	if len(f1) == 1 and len(f2) == 1:
		return "#N/A (scalar tensors)"
	if np.ptp(f1) == 0:
		return "#N/A (var(x) == 0 for pearsonr(x,y))"
	if np.ptp(f2) == 0:
		return "#N/A (var(y) == 0 for pearsonr(x,y))"
	if np.array_equal(f1, f2):
		return 1.0

	xm = f1 - f1.mean()
	ym = f2 - f2.mean()
	r = np.dot(xm / linalg.norm(xm), ym / linalg.norm(ym))
	return max(min(r, 1.0), -1.0)

def signed_ulp(raw_a, raw_b, fmt):
	"""Signed ULP distance (rank(a) - rank(b)) plus the valid mask, for same-format fp8."""
	rank = fp8_rank_table(fmt)
	ra = rank[raw_a].astype(np.int32)
	rb = rank[raw_b].astype(np.int32)
	ok = (ra != RANK_INVALID) & (rb != RANK_INVALID)
	sdiff = np.zeros(len(ra), dtype=np.int32)
	sdiff[ok] = ra[ok] - rb[ok]
	return (sdiff, ok)

def ulp_error(raw_a, raw_b, fmt):
	"""Max ULP distance between two same-format fp8 buffers."""
	sdiff, ok = signed_ulp(raw_a, raw_b, fmt)
	if not ok.any():
		return (None, None)
	diff = np.abs(sdiff)
	diff[~ok] = -1
	pos = int(np.argmax(diff))
	return (int(diff[pos]), pos)

def signed_ulp_error(raw_a, raw_b, fmt):
	"""Directional ULP stats: how far and which way f1 sits from f2 in code space."""
	sdiff, ok = signed_ulp(raw_a, raw_b, fmt)
	if not ok.any():
		return None
	d = sdiff[ok]
	return {
		"min": int(d.min()), "max": int(d.max()), "avg": float(d.mean()),
		"pos_cnt": int((d > 0).sum()), "neg_cnt": int((d < 0).sum()),
		"zero_cnt": int((d == 0).sum()), "n": int(d.size),
	}

def bit_diff(raw_a, raw_b, fmt=None):
	"""Per-bit flip counts between two fp8 byte buffers.

	fmt is only needed for the sign/exponent/mantissa field split; the raw bit
	counts and Hamming distance are format independent.
	"""
	x = np.bitwise_xor(raw_a, raw_b)
	bits = np.unpackbits(x.reshape(-1, 1), axis=1)	# column 0 is bit7 (the sign bit)
	per_bit = bits.sum(axis=0).astype(np.int64)[::-1]	# index by bit number, 0 = LSB
	out = {
		"per_bit": per_bit,
		"hamming": int(per_bit.sum()),
		"diff_bytes": int(np.count_nonzero(x)),
		"n": int(len(x)),
		"sign_mismatch": int(per_bit[7]),
	}
	if fmt is not None:
		ebits, mbits, bias, has_inf = FP8_FORMATS[fmt]
		e_mask = np.uint8((((1 << ebits) - 1) << mbits) & 0xFF)
		m_mask = np.uint8((1 << mbits) - 1)
		out["exp_mismatch"] = int(np.count_nonzero(np.bitwise_and(x, e_mask)))
		out["mant_mismatch"] = int(np.count_nonzero(np.bitwise_and(x, m_mask)))
		out["fields"] = (ebits, mbits)
	return out

def sign_diff(fa, fb):
	"""Value-level sign disagreement, ignoring the +0/-0 code split."""
	sa, sb = np.sign(fa), np.sign(fb)
	both_nz = (sa != 0) & (sb != 0)
	opposite = int(np.count_nonzero(both_nz & (sa != sb)))
	return (opposite, int(np.count_nonzero(both_nz)))

def show_fp8_table(fmt):
	tbl = fp8_value_table(fmt)
	rank = fp8_rank_table(fmt)
	ebits, mbits, bias, has_inf = FP8_FORMATS[fmt]
	finite = tbl[np.isfinite(tbl)]
	print("fp8 %s: exp_bits=%d mantissa_bits=%d bias=%d has_inf=%s" % (fmt, ebits, mbits, bias, has_inf))
	print("  max normal: %g, min subnormal: %g, finite codes: %d" %
		(np.max(finite), np.min(np.abs(finite[finite != 0])), len(finite)))
	for code in range(256):
		print("  0x%02X  %-14s rank=%s" %
			(code, repr(float(tbl[code])), rank[code] if rank[code] != RANK_INVALID else "-"))

def bin_fp8_err(args):
	data_fa, raw_a, dt1 = load_bin(args.f1, args.t1)
	data_fb, raw_b, dt2 = load_bin(args.f2, args.t2)
	print(args.f1, args.f2)

	if dt1 == dt2:
		print("Two files use the same dtype: ", dt1, dt2, ", shape: ", data_fa.shape, data_fb.shape)
		if not check_fsize(data_fa, data_fb):
			print("Two files size different, error")
			return 1
	else:
		print("Two files use different dtype: ", dt1, dt2, ", shape: ", data_fa.shape, data_fb.shape)
		if len(data_fa) != len(data_fb):
			print("Two files element count different: ", len(data_fa), len(data_fb), ", error")
			return 1

	if len(data_fa) == 0:
		print("Empty input, nothing to compare")
		return 1

	if (args.q1):
		data_fa = data_fa / pow(2, args.q1)
	if (args.q2):
		data_fb = data_fb / pow(2, args.q2)

	if (args.v):
		print("--------------", args.f1)
		print(data_fa)
		print("--------------", args.f2)
		print(data_fb)
		print("============")

	# fp8 carries NaN/Inf codes, keep them out of the metrics but report them
	nan_a, nan_b = int(np.isnan(data_fa).sum()), int(np.isnan(data_fb).sum())
	inf_a, inf_b = int(np.isinf(data_fa).sum()), int(np.isinf(data_fb).sum())
	if nan_a or nan_b or inf_a or inf_b:
		print("non-finite: f1 nan=%d inf=%d, f2 nan=%d inf=%d (excluded from metrics)" %
			(nan_a, inf_a, nan_b, inf_b))

	finite = np.isfinite(data_fa) & np.isfinite(data_fb)
	n_total = len(data_fa)
	if not finite.all():
		idx_map = np.nonzero(finite)[0]
		fa, fb = data_fa[finite], data_fb[finite]
	else:
		idx_map = None
		fa, fb = data_fa, data_fb

	if len(fa) == 0:
		print("No finite element pair to compare")
		return 1

	def orig_pos(p):
		return int(idx_map[p]) if idx_map is not None else int(p)
	return report(args, fa, fb, raw_a, raw_b, dt1, dt2, n_total, orig_pos)

def hex_at(raw, pos):
	return "" if raw is None else " (0x%02X)" % raw[pos]

def report(args, fa, fb, raw_a, raw_b, dt1, dt2, n_total, orig_pos):
	max_abs_err, p_abs = max_absolute_error(fa, fb)
	max_rel_err, p_rel = max_relative_error(fa, fb)
	avg_rel_err = avg_relative_error(fa, fb)
	cos_err = cosine_distance(fa, fb)
	pearson_err = pearson_correlation(fa, fb)
	psnr_err = psnr(fa, fb)

	o_abs, o_rel = orig_pos(p_abs), orig_pos(p_rel)
	print("max_abs_err:", max_abs_err, ", at position: ", o_abs,
		", left - right:", fa[p_abs], hex_at(raw_a, o_abs), "-", fb[p_abs], hex_at(raw_b, o_abs))
	print("max_rel_err:", max_rel_err, ", at position: ", o_rel,
		", left - right:", fa[p_rel], hex_at(raw_a, o_rel), "-", fb[p_rel], hex_at(raw_b, o_rel))
	print("avg_rel_err:", avg_rel_err)
	print("cosine_dist:", cos_err)
	print("pearson:", pearson_err)
	print("psnr:", psnr_err)

	# fp8 has only 256 codes, so exact-match and ULP stats are the useful accuracy view
	eq = int(np.count_nonzero(fa == fb))
	print("exact_match: %d/%d (%.4f%%), mismatch: %d" %
		(eq, len(fa), 100.0 * eq / len(fa), len(fa) - eq))

	# sign disagreement at value level works for any dtype pair
	opp, both_nz = sign_diff(fa, fb)
	print("sign_diff: %d/%d nonzero pairs with opposite sign (%.4f%%)" %
		(opp, both_nz, 100.0 * opp / both_nz if both_nz else 0.0))

	if raw_a is not None and raw_b is not None and dt1 == dt2:
		max_ulp, p_ulp = ulp_error(raw_a, raw_b, dt1)
		if max_ulp is not None:
			tbl = fp8_value_table(dt1)
			print("max_ulp_err: %d , at position: %d , left - right: %s (0x%02X) - %s (0x%02X)" %
				(max_ulp, p_ulp, tbl[raw_a[p_ulp]], raw_a[p_ulp], tbl[raw_b[p_ulp]], raw_b[p_ulp]))
		s = signed_ulp_error(raw_a, raw_b, dt1)
		if s is not None:
			print("signed_ulp: min %+d , max %+d , avg %+.4f (f1 higher: %d, lower: %d, equal: %d)" %
				(s["min"], s["max"], s["avg"], s["pos_cnt"], s["neg_cnt"], s["zero_cnt"]))
		same_bits = int(np.count_nonzero(raw_a == raw_b))
		print("identical_bytes: %d/%d (%.4f%%)" %
			(same_bits, len(raw_a), 100.0 * same_bits / len(raw_a)))

	if args.bit:
		if raw_a is None or raw_b is None:
			print("---- bit diff: skipped, needs both inputs in fp8 format ----")
		else:
			bd = bit_diff(raw_a, raw_b, dt1 if dt1 == dt2 else None)
			print("---- bit diff ----")
			print("  sign_bit_diff: %d/%d (%.4f%%)" %
				(bd["sign_mismatch"], bd["n"], 100.0 * bd["sign_mismatch"] / bd["n"]))
			if "fields" in bd:
				ebits, mbits = bd["fields"]
				print("  exp_diff:      %d/%d (%.4f%%)" %
					(bd["exp_mismatch"], bd["n"], 100.0 * bd["exp_mismatch"] / bd["n"]))
				print("  mant_diff:     %d/%d (%.4f%%)" %
					(bd["mant_mismatch"], bd["n"], 100.0 * bd["mant_mismatch"] / bd["n"]))
			else:
				print("  (different fp8 formats, no field split)")
				ebits = mbits = None
			print("  hamming: %d bits over %d bytes, avg %.4f bits/byte" %
				(bd["hamming"], bd["n"], bd["hamming"] / bd["n"]))
			for b in range(7, -1, -1):
				if ebits is None:
					tag = ""
				elif b == 7:
					tag = " sign"
				elif b >= mbits:
					tag = " exp"
				else:
					tag = " mant"
				print("    bit%d%-5s flips: %d (%.4f%%)" %
					(b, tag, bd["per_bit"][b], 100.0 * bd["per_bit"][b] / bd["n"]))

	if args.top:
		diff = np.abs(fa - fb)
		order = np.argsort(-diff)[:args.top]
		print("---- top %d abs diff ----" % min(args.top, len(order)))
		for p in order:
			o = orig_pos(p)
			print("  [%d] %s%s vs %s%s , abs=%g" %
				(o, fa[p], hex_at(raw_a, o), fb[p], hex_at(raw_b, o), diff[p]))
	return 0

def init_param(args):
	parser = argparse.ArgumentParser(
		description="Compare two binary file error in fp8 (E4M3/E5M2) format")
	parser.add_argument("-f1", type=str, required=False,
		help="input binary 1 filename")
	parser.add_argument("-f2", type=str, required=False,
		help="input binary 2 filename")
	parser.add_argument("-t1", type=str, required=False, default="e4m3",
		help="input binary 1 format: " + dtype_str)
	parser.add_argument("-t2", type=str, required=False, default="e4m3",
		help="input binary 2 format: " + dtype_str)
	parser.add_argument("-q1", type=int, required=False,
		help="Q value for quantized data 1")
	parser.add_argument("-q2", type=int, required=False,
		help="Q value for quantized data 2")
	parser.add_argument("-v", action='store_true', required=False,
		help="Show data in float")
	parser.add_argument("-top", type=int, required=False, default=0,
		help="Show top N elements with the largest absolute diff")
	parser.add_argument("-bit", action='store_true', required=False,
		help="Show per-bit flip counts (sign/exp/mantissa) between the two fp8 buffers")
	parser.add_argument("--show-table", type=str, required=False, dest="show_table",
		choices=sorted(FP8_FORMATS.keys()),
		help="Dump the 256 code value table of one fp8 format and exit")
	return parser.parse_args(args)

if __name__ == '__main__':
	# keep `| head` from spewing a BrokenPipeError traceback
	try:
		import signal
		signal.signal(signal.SIGPIPE, signal.SIG_DFL)
	except (ImportError, AttributeError):
		pass

	args = init_param(sys.argv[1:])
	if args.show_table:
		show_fp8_table(args.show_table)
		sys.exit(0)
	if not args.f1 or not args.f2:
		print("error: -f1 and -f2 are required (or use --show-table <fmt>)")
		sys.exit(2)
	for f, t in ((args.f1, args.t1), (args.f2, args.t2)):
		if not os.path.isfile(f):
			print("error: no such file:", f)
			sys.exit(2)
		if norm_dtype(t) not in FP8_FORMATS:
			try:
				np.dtype(norm_dtype(t))
			except TypeError:
				print("error: unsupported dtype:", t, ", expect", dtype_str)
				sys.exit(2)
	sys.exit(bin_fp8_err(args))
