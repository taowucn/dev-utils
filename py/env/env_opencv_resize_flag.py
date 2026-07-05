#!/usr/bin/env python3

import cv2 as cv

flags = [i for i in dir(cv) if i.startswith('INTER_')]
print( flags )
