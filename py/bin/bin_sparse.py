#!/usr/bin/env python3

import numpy as np
import argparse
import random

def create_sparse_file(opts):
    dk = np.fromfile(opts.kfile, dtype=np.int8)
    dk_size = dk.size
    density = opts.keepratio / 100.0
    zero_number = int(dk_size * (1 - density))
    print("Set %u int8 to 0, total %u, sparsity keep: %u" % (zero_number, dk_size, opts.keepratio))
    zero_list = [i for i in range(dk_size)]
    zero_list = random.sample(zero_list, zero_number)
    dk[zero_list] = 0
    dk.tofile(opts.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in", required=True, help='input file')
    parser.add_argument("-o", "--out", required=True, help='path to save output sparse file')
    parser.add_argument("-p", "--keepratio", type=int, default=50, help='keep ratio (0, 100)')
    opts = parser.parse_args()

    if opts.keepratio <= 0 or opts.keepratio >= 100:
        print("invalid keep ratio {}" % (opts.keepratio))

    create_sparse_file(opts)
