#!/usr/bin/env python3

import caffe
import cv2
import numpy as np

top_dir = './'

## Caffe model
model = top_dir + 'crnn_pure_lstm.prototxt'  	## TODO
weights = top_dir + 'crnn_pure_lstm.caffemodel' ## TODO

## Input data filename
data_path = top_dir + 'dra/1_1_512_71_fp32.bin' ## TODO
clip_path = top_dir + 'dra/clip.bin'            ## TODO
indicator_path = top_dir + 'dra/indicator.bin'  ## TODO
indicator2_path = top_dir + 'dra/indicator2.bin'## TODO

## Load model
net = caffe.Net(model, weights, caffe.TEST)

input_shape = net.blobs['data'].data.shape      ## TODO for data
data = np.fromfile(data_path, dtype=np.float32) ## TODO for float32
data = data.reshape(input_shape)

clip_shape = net.blobs['clip'].data.shape
clip = np.fromfile(clip_path, dtype=np.uint8)
clip = clip.reshape(clip_shape)

indicator_shape = net.blobs['noname_indicator26'].data.shape
indicator = np.fromfile(indicator_path, dtype=np.uint8)
indicator = indicator.reshape(indicator_shape)

indicator2_shape = net.blobs['noname_indicator35'].data.shape
indicator2 = np.fromfile(indicator2_path, dtype=np.uint8)
indicator2 = indicator2.reshape(indicator2_shape)

## Load input data
net.blobs['data'].data[...] = data				## TODO for input data name
net.blobs['clip'].data[...] = clip
net.blobs['noname_indicator26'].data[...] = indicator
net.blobs['noname_indicator35'].data[...] = indicator2

## inference model
outs = net.forward()

## save output
outs['view_blob24'].astype(np.float32).tofile('pc_out/' + 'view_blob24_fp32_71x1.bin')  ## TODO for output name