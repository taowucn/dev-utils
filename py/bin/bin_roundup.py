#!/usr/bin/env python3

import argparse
import sys


def round_up(org_size: int, align_size: int) -> int:
    if org_size < 0 or align_size <= 0:
        raise ValueError("size, align_size must bigger than zero: ", org_size, align_size)

    f_sz = org_size / align_size
    ret_sz = ((org_size + align_size - 1) // align_size) * align_size
    print(f"float {org_size} / {align_size} = {f_sz}")
    print(f"round_up({org_size} , {align_size}) = {ret_sz}")
    return ret_sz


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Round up a size to the nearest multiple of align_size."
    )
    parser.add_argument("size", type=int, help="the size to round up (>= 0)")
    parser.add_argument("align_size", type=int, help="the alignment size (> 0)")
    args = parser.parse_args(argv)

    try:
        round_up(args.size, args.align_size)
    except ValueError as err:
        parser.error(" ".join(str(a) for a in err.args))
    return 0


if __name__ == "__main__":
    sys.exit(main())

