#/usr/bin/env python3
import os, sys, argparse
import numpy as np

a=np.arange(16).reshape(2,2,4)
print(a)
b=a.transpose(1,0,2)
print(b)

a.tofile("a.bin")
b.tofile("b.bin")
