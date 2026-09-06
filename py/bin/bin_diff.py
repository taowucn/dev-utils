#!/usr/bin/env python3

import os
import sys

BUF_SIZE = 64 * 1024
PAIR_TABLE_SIZE = 256 * 256
MAX_PAIR_LINES = 32


def bit_diff_count(left_data, right_data):
    return (left_data ^ right_data).bit_count()


class DiffStat:
    def __init__(self):
        self.bytes_compared = 0
        self.diff_bytes = 0
        self.diff_bits = 0
        self.diff_runs = 0
        self.first_offset = 0
        self.last_offset = 0
        self.printed = 0
        self.max_print = 0
        self.truncated = False
        self.pair_count = None
        self.pair_kinds = 0

    def record_diff(self, offset, left_data, right_data, in_run):
        if self.diff_bytes == 0:
            self.first_offset = offset
        self.last_offset = offset
        self.diff_bytes += 1
        self.diff_bits += bit_diff_count(left_data, right_data)
        if not in_run:
            self.diff_runs += 1

        if self.pair_count is not None:
            index = ((left_data << 8) | right_data) & 0xFFFF
            slot = self.pair_count[index]
            if slot == 0:
                self.pair_kinds += 1
            self.pair_count[index] = slot + 1

        if self.max_print != 0 and self.printed >= self.max_print:
            if not self.truncated:
                self.truncated = True
                print("... more differences are counted but not printed")
            return

        print(
            f"[{self.diff_bytes - 1:08d}] byte_offset at: 0x{offset:08x}, "
            f"left: 0x{left_data:02x}, right: 0x{right_data:02x}"
        )
        self.printed += 1


def get_file_size(path):
    try:
        return os.path.getsize(path)
    except OSError as e:
        print(f"{path}: {e}")
        return None


def compare_files(left_file, right_file, common_size, stat):
    left_fp = None
    right_fp = None
    left_buf = None
    right_buf = None
    offset = 0
    in_run = False

    try:
        left_fp = open(left_file, "rb")
        right_fp = open(right_file, "rb")

        while offset < common_size:
            remain = common_size - offset
            want = min(remain, BUF_SIZE)

            left_buf = left_fp.read(want)
            if len(left_buf) != want:
                print(f"{left_file}: read failed at offset 0x{offset:08x}")
                return -1

            right_buf = right_fp.read(want)
            if len(right_buf) != want:
                print(f"{right_file}: read failed at offset 0x{offset:08x}")
                return -1

            if left_buf != right_buf:
                for i in range(want):
                    if left_buf[i] != right_buf[i]:
                        stat.record_diff(offset + i, left_buf[i], right_buf[i], in_run)
                        in_run = True
                    else:
                        in_run = False
            else:
                in_run = False

            offset += want
            stat.bytes_compared = offset
    finally:
        if right_fp is not None:
            right_fp.close()
        if left_fp is not None:
            left_fp.close()

    return 0


def print_pair_table(stat):
    if stat.pair_count is None or stat.pair_kinds == 0:
        return

    used = []
    for i in range(PAIR_TABLE_SIZE):
        if stat.pair_count[i] != 0:
            used.append(i)

    if not used:
        return

    used.sort(key=lambda idx: stat.pair_count[idx], reverse=True)
    limit = min(len(used), MAX_PAIR_LINES)

    print("---------- Byte difference count ----------")
    print("  left -> right :      count   bits  percent")
    for i in range(limit):
        index = used[i]
        left_data = (index >> 8) & 0xFF
        right_data = index & 0xFF
        count = stat.pair_count[index]
        percent = (count * 100.0) / stat.diff_bytes if stat.diff_bytes else 0.0
        print(
            f"  0x{left_data:02x} -> 0x{right_data:02x} : {count:10d}   "
            f"{bit_diff_count(left_data, right_data):4d}  {percent:6.2f}%"
        )

    if len(used) > limit:
        print(f"  ... {len(used) - limit} more kinds not listed")


def print_summary(left_file, right_file, left_size, right_size, stat):
    percent = 0.0
    if stat.bytes_compared != 0:
        percent = (stat.diff_bytes * 100.0) / stat.bytes_compared

    print("---------------- Summary ----------------")
    print(f"Left  : {left_file} ({left_size} bytes)")
    print(f"Right : {right_file} ({right_size} bytes)")
    print(f"Bytes compared : {stat.bytes_compared}")
    print(f"Bytes different: {stat.diff_bytes} ({percent:.4f}%)")
    print(f"Bits different : {stat.diff_bits}")
    print(f"Diff byte kinds: {stat.pair_kinds}")

    if stat.diff_bytes != 0:
        print(f"First diff at  : 0x{stat.first_offset:08x} ({stat.first_offset})")
        print(f"Last diff at   : 0x{stat.last_offset:08x} ({stat.last_offset})")

    if left_size != right_size:
        extra = left_size - right_size if left_size > right_size else right_size - left_size
        who = left_file if left_size > right_size else right_file
        print(f"Size mismatch  : {who} has {extra} extra bytes")

    if stat.truncated:
        print(f"Printed        : {stat.printed} of {stat.diff_bytes} diff lines")
    print("-----------------------------------------")

    print_pair_table(stat)

    if stat.diff_bytes == 0 and left_size == right_size:
        print("Match")
    else:
        print("Mismatch")


def main(argv):
    if len(argv) not in (3, 4):
        print("Version: 0.0.2")
        print(f"Usage: {argv[0]} <Left filename> <Right filename>")
        print(f"       {argv[0]} <Left filename> <Right filename> <Max print lines>")
        return 2

    left_file = argv[1]
    right_file = argv[2]

    left_size = get_file_size(left_file)
    if left_size is None:
        return 2
    right_size = get_file_size(right_file)
    if right_size is None:
        return 2

    stat = DiffStat()
    if len(argv) == 4:
        try:
            stat.max_print = int(argv[3], 0)
        except ValueError:
            print(f"Invalid max print value: {argv[3]}")
            return 2

    if left_size == 0 or right_size == 0:
        print(f"File size is zero, {left_file}: {left_size},  {right_file}: {right_size}")
        return 2

    stat.pair_count = [0] * PAIR_TABLE_SIZE
    if left_size != right_size:
        print(f"File size mismatch, {left_file}: {left_size},  {right_file}: {right_size}")
        print(f"Comparing the common {min(left_size, right_size)} bytes ...")

    common_size = min(left_size, right_size)
    rc = compare_files(left_file, right_file, common_size, stat)
    if rc < 0:
        return 2

    print_summary(left_file, right_file, left_size, right_size, stat)
    return 1 if (stat.diff_bytes != 0 or left_size != right_size) else 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(130)
