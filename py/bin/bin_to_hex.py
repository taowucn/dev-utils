#!/usr/bin/env python

import sys
import os
import argparse


def process_bin_file(input_path: str,
                     output_path: str = None,
                     bytes_per_line: int = 16,
                     swap: bool = True) -> None:
    """
    Convert a binary BIN file into a text HEX file with a fixed number of bytes
    per line (16B by default, the same row width as `hexdump -C`).
    Conversion rule: split into bytes_per_line chunks -> reverse the bytes within
    each row (MSB/LSB swap) -> join as lowercase HEX.
    Default output name: <input file name>.txt
    :param input_path: input BIN file path
    :param output_path: output text file path (optional)
    :param bytes_per_line: bytes per output line
    :param swap: reverse bytes within each row (keeps the original order if disabled)
    """
    if bytes_per_line <= 0:
        raise ValueError(f"bytes per line must be positive (got {bytes_per_line})")

    if output_path is None:
        output_path = f"{input_path}.txt"

    total_bytes = 0
    line_count = 0
    tail_bytes = 0

    try:
        with open(input_path, 'rb') as bin_f, \
             open(output_path, 'w', encoding='utf-8') as txt_f:
            while True:
                chunk = bin_f.read(bytes_per_line)
                if not chunk:
                    break

                total_bytes += len(chunk)
                line_count += 1

                # The last row may be shorter than bytes_per_line; handle its actual length
                if len(chunk) != bytes_per_line:
                    tail_bytes = len(chunk)

                # Core logic: reverse the bytes within the row -> lowercase HEX
                line_bytes = chunk[::-1] if swap else chunk
                txt_f.write(line_bytes.hex())
                txt_f.write('\n')

        print(f"✅ Conversion complete!")
        print(f"📥 Input file: {input_path}")
        print(f"📤 Output file: {output_path}")
        print(f"🔢 Total bytes read: {total_bytes}")
        print(f"📏 Output lines: {line_count} ({bytes_per_line} bytes per line, "
              f"byte swap: {'on' if swap else 'off'})")
        if tail_bytes:
            print(f"⚠️  Last row is shorter than {bytes_per_line} bytes, actual {tail_bytes} bytes")

    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Input file not found: {input_path}")
    except IsADirectoryError:
        raise IsADirectoryError(f"❌ Input path is a directory: {input_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BIN to HEX text tool: N bytes per line, bytes reversed within each row "
                    "(the inverse of hex_to_bin.py)")
    parser.add_argument("input", help="input BIN file path")
    parser.add_argument("output", nargs='?', default=None,
                        help="output text file path (default: <input file name>.txt)")
    parser.add_argument("-n", "--bytes-per-line", type=int, default=16,
                        help="bytes per line (default: 16)")
    parser.add_argument("--no-swap", action="store_true",
                        help="do not reverse bytes within a row, keep the original order")
    args = parser.parse_args()

    try:
        process_bin_file(args.input, args.output,
                         bytes_per_line=args.bytes_per_line,
                         swap=not args.no_swap)
    except Exception as e:
        print(e)
        sys.exit(1)
