#!/usr/bin/env python3

import os, sys, argparse
import numpy as np

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8> "

dtype_ext_str = "<bf16|fp8_e4m3|fp8_e5m2> "
EXT_DTYPES = {"bf16", "fp8_e4m3", "fp8_e5m2"}
dconv_fpx = None

# --------------------------- fp8 -> fp32 ---------------------------
def parse_fp8_e4m3(byte_val: int) -> float:
	"""
	解析一个 8 位整数（0-255）为 FP8 E4M3 格式的浮点数。

	Args:
		byte_val: 8 位无符号整数，表示 FP8 E4M3 的二进制编码

	Returns:
		解析后的 Python float (双精度)
	"""
	# 确保输入在 0-255 范围内
	byte_val = byte_val & 0xFF

	# 1. 提取各字段
	sign = (byte_val >> 7) & 0x1       # 第7位：符号位
	exp  = (byte_val >> 3) & 0xF       # 第6-3位：指数位 (4位)
	mant = byte_val & 0x7              # 第2-0位：尾数位 (3位)

	bias = 7  # E4M3 的指数偏移量

	# 2. 根据指数值分类处理
	if exp == 0:
		# 亚规格数 (denormal)：隐含前导 0，指数 = 1 - bias
		value = (mant / 8.0) * (2 ** (1 - bias))
	elif exp == 15:  # 0b1111
		# E4M3 特殊规则：指数全1不表示 Inf/NaN，而表示最大有限值
		# 尾数全1时为最大值 448.0 (1.111 * 2^8 = 15/8 * 256 = 480? 需注意具体规范)
		# 按 OpenAI / NVIDIA 常见定义：exp=15, mant=7 -> NaN; 其他情况 Inf
		# 但 E4M3 通常是 "有限最大" 模式，这里按常见 AI 框架实现：
		value = (1.0 + mant / 8.0) * (2 ** (14 - bias))  # 用 exp=14 的最大值表示？
		# 更准确的方式：许多实现中 exp=1111 保留，这里我们当作最大正规数处理
		# 实际上标准 E4M3 最大正值是 0 1111 111 = 448.0
		if mant == 7:
			# 全1有时被定义为 NaN，但 E4M3 通常不用 NaN
			value = float('nan')
		else:
			value = (1.0 + mant / 8.0) * (2 ** (exp - bias))
	else:
		# 普通规格数：隐含前导 1
		value = (1.0 + mant / 8.0) * (2 ** (exp - bias))

	# 3. 应用符号
	if sign:
		value = -value

	return value

def parse_fp8_e5m2(byte_val: int) -> float:
	"""
	解析一个 8 位整数为 FP8 E5M2 格式的浮点数。

	Args:
		byte_val: 8 位无符号整数 (0-255)，表示 FP8 E5M2 的二进制编码

	Returns:
		解析后的 Python float
	"""
	byte_val = byte_val & 0xFF

	# 1. 提取字段
	sign = (byte_val >> 7) & 0x1    # 第7位：符号位
	exp  = (byte_val >> 2) & 0x1F # 第6-2位：指数位 (5位)
	mant = byte_val & 0x3           # 第1-0位：尾数位 (2位)

	bias = 15  # E5M2 的指数偏移量

	# 2. 分类处理
	if exp == 0:
		# 亚规格数 (denormal)：隐含前导 0，指数 = 1 - bias = -14
		value = (mant / 4.0) * (2 ** (1 - bias))
	elif exp == 31:  # 0b11111，指数全1
		if mant == 0:
			# 尾数全0 -> 无穷大 (Inf)
			value = float('inf')
		else:
			# 尾数非0 -> 非数 (NaN)
			value = float('nan')
	else:
		# 普通规格数：隐含前导 1
		value = (1.0 + mant / 4.0) * (2 ** (exp - bias))

	# 3. 应用符号
	if sign:
		# NaN 不区分正负，但保持语义上 sign 不影响 NaN 的 "nan 表示
		if value == float('inf'):
			value = float('-inf')
		elif value != value:  # 检测 NaN
			pass  # NaN 符号位可忽略
		else:
			value = -value

	return value

# ------------------------------------------------------

def test_fp8_e4m3():
	test_cases = [
		0x00,  # 0.0
		0x40,  # 1.0
		0x3C,  # 0.5
		0x38,  # 0.25
		0x7B,  # 最大正规数
		0x7C,  # Inf
		0x7F,  # NaN
		0x80,  # -0.0
		0xFC,  # -Inf
		0x01,  # 最小亚规格正数
	]

	print("FP8 E4M3 测试:")
	print("-" * 40)
	for b in test_cases:
		f = parse_fp8_e4m3(b)
		print(f"0x{b:02X} (0b{b:08b}) -> {f}")

def test_fp8_e5m2():
	test_cases = [
		0x00,  # 0.0
		0x40,  # 1.0
		0x3C,  # 0.5
		0x38,  # 0.25
		0x7B,  # 最大正规数
		0x7C,  # Inf
		0x7F,  # NaN
		0x80,  # -0.0
		0xFC,  # -Inf
		0x01,  # 最小亚规格正数
	]

	print("FP8 E5M2 测试:")
	print("-" * 40)
	for b in test_cases:
		f = parse_fp8_e5m2(b)
		print(f"0x{b:02X} (0b{b:08b}) -> {f}")

# ---------- Statistics ----------

def statis_data(args, data_x_f):
	if args.s:
		max_value = np.max(data_x_f)
		min_value = np.min(data_x_f)
		mean_value = np.mean(data_x_f)
		median_value = np.median(data_x_f)
		diff_value = max_value - min_value
		diff_range = np.ceil(diff_value).astype(np.int32)

		print("max_value:", max_value)
		print("min_value:", min_value)
		print("diff_value:", diff_value, "diff_range:", diff_range)
		print("mean_value:", mean_value)
		print("median_value:", median_value)

# ---------- Main viewer ----------

def bin_view(args):
	filepath = args.i
	data_type = args.f

	is_fp_ext = False
	if data_type in EXT_DTYPES:
		is_fp_ext = True

	# 1. Load data
	if data_type == "bf16":
		## Don't support
		# NumPy >= 1.20 supports bfloat16 natively
		raise NotImplementedError("BF16 support is not implemented in this viewer.")
	elif data_type == "fp8_e4m3":
		raw = np.fromfile(filepath, dtype=np.uint8)
		data = np.vectorize(parse_fp8_e4m3, otypes=[np.float32])(raw)
		print(f"dtype: {data_type} (displayed as float32), shape:", data.shape)
	elif data_type == "fp8_e5m2":
		raw = np.fromfile(filepath, dtype=np.uint8)
		data = np.vectorize(parse_fp8_e5m2, otypes=[np.float32])(raw)
		print(f"dtype: {data_type} (displayed as float32), shape:", data.shape)
	else:
		data = np.fromfile(filepath, dtype=data_type)
		print(f"dtype: {data_type}, shape:", data.shape)

	# 2. Reshape by width
	if args.w:
		num = data.shape[0]
		height = int(num / args.w)
		print(num, " Reshape to ", height, args.w)
		data = data.reshape(height, args.w)

	# 3. Optional Q-format dequant (works on float interpretation)
	if args.q is not None:
		data_f = data.astype(np.float32)
		data_f = data_f * pow(2, -(args.q))
		if args.v:
			for row in data_f:
				print(row)
		else:
			print(data_f)
			statis_data(args, data_f)
	else:
		if args.v:
			for row in data:
				print(row)
		else:
			print(data)
			statis_data(args, data)

	# 4. Threshold filtering
	#    For comparison we always use a float32 view
	compare_data = data.astype(np.float32) if (is_fp_ext) else data

	threshold_max = args.a
	if threshold_max is not None:
		filter_data = np.where(compare_data > threshold_max)
		filter_data = np.asarray(filter_data)
		print(f"\n---------------- show bigger than threshold.Max: {threshold_max}")
		for idx in np.nditer(filter_data):
			print("idx: {}, data: {}".format(idx, data[idx]))
		print("data bigger than threshold.Max {} shape: {}".format(threshold_max, filter_data.shape))

	threshold_min = args.b
	if threshold_min is not None:
		filter_data = np.where(compare_data < threshold_min)
		filter_data = np.asarray(filter_data)
		print(f"\n---------------- show smaller than threshold.Min: {threshold_min}")
		for idx in np.nditer(filter_data):
			print("idx: {}, data: {}".format(idx, data[idx]))
		print("data smaller than threshold.Min {} shape: {}".format(threshold_min, filter_data.shape))


# ---------- Argument parsing ----------

def init_param(args):
	parser = argparse.ArgumentParser(description="View binary file with specific format "
								  "(1.0.0), support bf16,fp8")
	parser.add_argument("-i", type=str, required=True, default="input.bin",
		help="input binary filename")
	parser.add_argument("-f", type=str, required=False, default="float32",
		help="input binary format: " + dtype_str + dtype_ext_str)
	parser.add_argument("-q", type=int, required=False,
		help="Q value for quantized data")
	parser.add_argument("-w", type=int, required=False,
		help="Show element number in row (width)")
	parser.add_argument("-v", action='store_true', required=False,
		help="Show data by print txt")

	parser.add_argument("-a", type=float, required=False,
		help="threshold.Max, show data when data > threshold.Max")
	parser.add_argument("-b", type=float, required=False,
		help="threshold.Min, show data when data < threshold.Min")

	parser.add_argument("-s", action='store_true', required=False, default=False,
		help="statis data (max, min, avg, etc.)")

	parser.add_argument("-W", type=int, required=False, default=10,
		help="image width")
	parser.add_argument("-H", type=int, required=False, default=1,
		help="image height")
	parser.add_argument("-C", type=int, required=False, default=1,
		help="image channel")
	parser.add_argument("-coord", type=str, required=False,
		help="coordinate, format(x, y) at (w, h)")

	return parser.parse_args(args)


if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_view(args)
