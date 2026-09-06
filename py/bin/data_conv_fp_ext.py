#!/usr/bin/env python3

import struct
import numpy as np

# --------------------------- fp8 -> fp32 ---------------------------

def fp8_e4m3_to_float32(byte_val: int) -> float:
	"""Convert an FP8 E4M3 value (stored in uint8) to Python float."""
	sign = (byte_val >> 7) & 0x1
	exp = (byte_val >> 3) & 0xF
	mant = byte_val & 0x7

	if exp == 0xF:
		if mant == 0:
			return float('-inf') if sign else float('inf')
		return float('nan')

	if exp == 0:
		if mant == 0:
			return -0.0 if sign else 0.0
		val = (mant / 8.0) * (2 ** (1 - 7))
	else:
		val = (1.0 + mant / 8.0) * (2 ** (exp - 7))

	return -val if sign else val


def fp8_e5m2_to_float32(byte_val: int) -> float:
	"""Convert an FP8 E5M2 value (stored in uint8) to Python float."""
	sign = (byte_val >> 7) & 0x1
	exp = (byte_val >> 2) & 0x1F
	mant = byte_val & 0x3

	if exp == 0x1F:
		if mant == 0:
			return float('-inf') if sign else float('inf')
		return float('nan')

	if exp == 0:
		if mant == 0:
			return -0.0 if sign else 0.0
		val = (mant / 4.0) * (2 ** (1 - 15))
	else:
		val = (1.0 + mant / 4.0) * (2 ** (exp - 15))

	return -val if sign else val


def load_file_fp8_as_float32(filepath: str, fp8_type: str) -> np.ndarray:
	"""Read a raw FP8 binary file and return a float32 array."""
	raw = np.fromfile(filepath, dtype=np.uint8)
	if fp8_type == "fp8_e4m3":
		conv = np.vectorize(fp8_e4m3_to_float32, otypes=[np.float32])
	elif fp8_type == "fp8_e5m2":
		conv = np.vectorize(fp8_e5m2_to_float32, otypes=[np.float32])
	else:
		raise ValueError(f"Unsupported FP8 type: {fp8_type}")
	return conv(raw).astype(np.float32)

# --------------------------- fp32 -> fp8 ---------------------------

def _float32_to_fp8(value: float, exponent_bits: int, mantissa_bits: int, bias: int) -> int:
	"""Round a float32 value to an IEEE-like FP8 encoding."""
	value = np.float32(value)
	max_exponent = (1 << exponent_bits) - 1
	max_mantissa = (1 << mantissa_bits) - 1
	if np.isnan(value):
		return (max_exponent << mantissa_bits) | max_mantissa

	sign = 0x80 if np.signbit(value) else 0
	abs_value = abs(float(value))
	if np.isinf(value):
		return sign | (max_exponent << mantissa_bits)
	if abs_value == 0.0:
		return sign

	finite_codes = []
	finite_values = []
	for exponent in range(max_exponent):
		for mantissa in range(max_mantissa + 1):
			if exponent == 0:
				finite_value = (mantissa / (1 << mantissa_bits)) * 2.0 ** (1 - bias)
			else:
				finite_value = (1.0 + mantissa / (1 << mantissa_bits)) * 2.0 ** (exponent - bias)
			finite_codes.append((exponent << mantissa_bits) | mantissa)
			finite_values.append(finite_value)

	if abs_value >= finite_values[-1]:
		return sign | finite_codes[-1]
	nearest_index = min(
		range(len(finite_values)),
		key=lambda index: (abs(finite_values[index] - abs_value), finite_codes[index] & 1),
	)
	return sign | finite_codes[nearest_index]


def float32_to_fp8_e4m3(value: float) -> int:
	return _float32_to_fp8(value, exponent_bits=4, mantissa_bits=3, bias=7)

def float32_to_fp8_e5m2(value: float) -> int:
	return _float32_to_fp8(value, exponent_bits=5, mantissa_bits=2, bias=15)

# --------------------------------------------------------------------


# --------------------------- fp32 -> bf16 ---------------------------
def float32_to_bf16(value):
	"""Convert a float32 value to its BF16 uint16 representation."""
	if np.isnan(value):
		return 0x7FC0
	if np.isinf(value):
		return 0x7F80 if value > 0 else 0xFF80
	if value == 0:
		return 0x0000 if np.signbit(value) == 0 else 0x8000

	packed = struct.pack('>f', value)
	uint32_value = struct.unpack('>I', packed)[0]
	sign = (uint32_value >> 31) & 0x1
	exponent = (uint32_value >> 23) & 0xFF
	mantissa = (uint32_value >> 16) & 0x7F

	if exponent == 0xFF:
		return 0x7F80 | (sign << 15) | (1 if mantissa != 0 else 0)

	return ((sign << 15) | (exponent << 7) | mantissa) & 0xFFFF

# --------------------------- bf16 -> fp32 ---------------------------

def bf16_to_float32(bf16_value):
	"""Convert a BF16 uint16 representation to float32."""
	bf16_value = int(bf16_value)
	sign = (bf16_value >> 15) & 0x1
	exponent = (bf16_value >> 7) & 0xFF
	mantissa = bf16_value & 0x7F

	if exponent == 0xFF:
		if mantissa == 0:
			return float('inf') if sign == 0 else float('-inf')
		return float('nan')

	float32_value = (sign << 31) | (exponent << 23) | (mantissa << 16)
	packed = struct.pack('>I', float32_value)
	return struct.unpack('>f', packed)[0]


def load_file_bf16_as_float32(filepath: str, bf16_type: str) -> np.ndarray:
	"""Read a raw BF16 binary file and return a float32 array."""
	raw = np.fromfile(filepath, dtype=np.uint16)
	if bf16_type != "bf16":
		raise ValueError(f"Unsupported BF16 type: {bf16_type}")
	return np.vectorize(bf16_to_float32, otypes=[np.float32])(raw).astype(np.float32)


# ------------------------ float16 <-> bf16 -------------------------

def float16_to_bf16(value):
	"""Convert a float16 value to its BF16 uint16 representation."""
	return float32_to_bf16(np.float32(value))


def bf16_to_float16(bf16_value):
	"""Convert a BF16 uint16 representation to a NumPy float16 value."""
	return np.float16(bf16_to_float32(bf16_value))

# -----------------------------------------------------
