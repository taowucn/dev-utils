#!/usr/bin/env python3

import os, sys, argparse

CHUNK_SIZE = 1024 * 1024

def parse_int(text):
	## accept decimal (4096), hex (0x1000), octal (0o17) and binary (0b101) literals
	try:
		return int(text, 0)
	except ValueError:
		raise argparse.ArgumentTypeError("invalid integer value: '{}'".format(text))


def bin_extract(args):
	file_size = os.path.getsize(args.i)
	offset = args.s
	size = args.n

	print("input file:", args.i)
	print("input size:", file_size, "(0x{:x})".format(file_size))
	print("-"*60)

	if offset < 0:
		raise ValueError("offset must not be negative: {}".format(offset))
	if offset > file_size:
		raise ValueError("offset {} (0x{:x}) is beyond the end of file {} (0x{:x})".format(
			offset, offset, file_size, file_size))

	rest = file_size - offset
	if size is None:
		## no size specified, extract until the end of file
		size = rest
	elif size < 0:
		raise ValueError("size must not be negative: {}".format(size))
	elif size > rest:
		print("[Warning] size {} (0x{:x}) exceeds the end of file, clamp to {} (0x{:x})".format(
			size, size, rest, rest))
		size = rest

	print("offset: {} (0x{:x})".format(offset, offset))
	print("size:   {} (0x{:x})".format(size, size))
	print("range:  [0x{:x}, 0x{:x})".format(offset, offset + size))

	written = 0
	with open(args.i, 'rb') as src, open(args.o, 'wb') as dst:
		src.seek(offset)
		while written < size:
			chunk = src.read(min(CHUNK_SIZE, size - written))
			if not chunk:
				break
			dst.write(chunk)
			written += len(chunk)

	if written != size:
		raise IOError("short read: expect {} bytes, got {} bytes".format(size, written))

	print("-"*60)
	print('Extract File: {} -> {}'.format(args.i, args.o))


def init_param(args):
	parser = argparse.ArgumentParser(
		description="Extract a range (offset, size) from a binary file")
	parser.add_argument("-i", type=str, required=True,
		help="input binary filename")
	parser.add_argument("-o", type=str, required=True,
		help="ouput binary filename")

	parser.add_argument("-s", type=parse_int, required=False, default=0,
		help="start offset in bytes, decimal or hex (default: 0)")
	parser.add_argument("-n", type=parse_int, required=False, default=None,
		help="extract size in bytes, decimal or hex (default: until the end of file)")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	try:
		bin_extract(args)
	except FileNotFoundError:
		print("[Error] input file not found: {}".format(args.i))
		sys.exit(1)
	except (ValueError, IOError) as err:
		print("[Error]", err)
		sys.exit(1)
