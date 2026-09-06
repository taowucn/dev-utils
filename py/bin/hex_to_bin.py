#!/usr/bin/env python

import sys
import os
import argparse


def process_hex_file(input_path: str,
                     output_path: str = None,
                     bytes_per_line: int = 64,
                     swap: bool = True) -> None:
    """
    Convert a text HEX file into a binary BIN file, reversing the 2-char (byte)
    groups of every line (128 hex chars / 64 bytes per line by default).
    Grouping rule: split each line into 2-char groups in order -> reverse the
    group list -> join and decode via fromhex.
    Default output name: <input file name>.bin2
    :param input_path: input HEX file path
    :param output_path: output BIN file path (optional)
    :param bytes_per_line: bytes encoded per input line (2 hex chars per byte)
    :param swap: reverse the byte groups within each line (keeps the original order if disabled)
    """
    if bytes_per_line <= 0:
        raise ValueError(f"bytes per line must be positive (got {bytes_per_line})")

    # Each byte takes 2 hex chars, so a line is bytes_per_line * 2 chars wide
    expected_chars = bytes_per_line * 2

    if output_path is None:
        output_path = f"{input_path}.bin2"

    all_binary_data = bytearray()

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            line_num = 0
            for line in f:
                line_num += 1
                stripped_line = line.strip()

                # Skip empty lines
                if not stripped_line:
                    print(f"⚠️  Line {line_num}: empty line, skipped")
                    continue

                # Every line must be exactly expected_chars characters long
                if len(stripped_line) != expected_chars:
                    raise ValueError(f"Line {line_num}: unexpected length "
                                     f"(expected {expected_chars}, got {len(stripped_line)})")

                # Filter out non-hex characters
                valid_chars = ''.join([c for c in stripped_line if c in '0123456789abcdefABCDEF'])
                if len(valid_chars) != expected_chars:
                    print(f"⚠️  Line {line_num}: length changed after filtering non-HEX characters, corrected to {len(valid_chars)}")
                    if len(valid_chars) % 2 != 0:
                        valid_chars += '0'
                        print(f"⚠️  Line {line_num}: length is {len(valid_chars)} after zero-padding")

                # Core logic: split into 2-char groups -> reverse the groups -> join
                groups = [valid_chars[i:i+2] for i in range(0, len(valid_chars), 2)]
                ordered_groups = groups[::-1] if swap else groups  # reversed group list
                target_hex = ''.join(ordered_groups)

                # Convert to binary bytes and append to the accumulated data
                line_bytes = bytes.fromhex(target_hex)
                all_binary_data.extend(line_bytes)

        # Write the BIN file
        with open(output_path, 'wb') as bin_f:
            bin_f.write(all_binary_data)

        print(f"✅ Conversion complete!")
        print(f"📥 Input file: {input_path}")
        print(f"📤 Output file: {output_path}")
        print(f"🔢 Total bytes converted: {len(all_binary_data)}")
        print(f"📏 Line width: {expected_chars} hex chars ({bytes_per_line} bytes per line, "
              f"byte swap: {'on' if swap else 'off'})")

    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Input file not found: {input_path}")
    except ValueError as e:
        raise ValueError(f"❌ Data format error: {str(e)}")
    except Exception as e:
        raise Exception(f"❌ Unknown error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HEX text to BIN tool: N bytes per line, byte groups reversed within "
                    "each line (the inverse of bin_to_hex.py)",
        epilog="examples:\n"
               "  %(prog)s test.dat                     # -> test.dat.bin2\n"
               "  %(prog)s test.dat out.bin             # explicit output name\n"
               "  %(prog)s test.dat -n 16               # 32 hex chars per line\n"
               "  %(prog)s test.dat --no-swap           # keep the original byte order",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="input HEX text file path")
    parser.add_argument("output", nargs='?', default=None,
                        help="output BIN file path (default: <input file name>.bin2)")
    parser.add_argument("-n", "--bytes-per-line", type=int, default=64,
                        help="bytes encoded per line, 2 hex chars each (default: 64, "
                             "i.e. 128 hex chars per line)")
    parser.add_argument("--no-swap", action="store_true",
                        help="do not reverse the byte groups within a line, keep the original order")
    args = parser.parse_args()

    try:
        process_hex_file(args.input, args.output,
                         bytes_per_line=args.bytes_per_line,
                         swap=not args.no_swap)
    except Exception as e:
        print(e)
        sys.exit(1)
