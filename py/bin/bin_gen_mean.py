import numpy as np

Q = 0
ht = 608
wd = 800

dt = np.zeros((3,ht,wd), np.uint8)
dt[0,:,:] = 103
dt[1,:,:] = 116
dt[2,:,:] = 123
dt = dt/pow(2,Q)
dt.tofile("mean.bin")
