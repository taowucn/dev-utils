#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

#@save
def use_svg_display():
    """使用svg格式在Jupyter中显示绘图"""
    #backend_inline.set_matplotlib_formats('svg')

#@save
def set_figsize(figsize=(10, 10)):
    """设置matplotlib的图表大小"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(20, 20), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

def bin_plot(args):
	data_x = np.fromfile(args.x, dtype=args.fx)
	data_y = np.fromfile(args.y, dtype=args.fy)

	if (args.qx):
		data_x_f = data_x.astype(np.float32)
		data_x_f = data_x_f/pow(2, args.qx)
	else:
		data_x_f = data_x

	if (args.qy):
		data_y_f = data_x.astype(np.float32)
		data_y_f = data_y_f/pow(2, args.qx)
	else:
		data_y_f = data_y

	if (args.s):
		print("cut data start [{} :]".format(args.s))
		data_x_f = data_x_f[args.s:]
		data_y_f = data_y_f[args.s:]

	if (args.e):
		print("cut data end [: {}]".format(args.e))
		data_x_f = data_x_f[:args.e]
		data_y_f = data_y_f[:args.e]

	if (args.v):
		print("--- x, y data in float32 ---")
		print("x:", data_x_f)
		print("y:", data_y_f)

	plot(data_x_f, data_y_f, 'x', 'y', legend=['xy'])

	if (args.o):
		print("save image as:", args.o)
		plt.savefig(args.o)
	else:
		print("show image in live")
		plt.show()

## Test
'''
def f(x):
    return 3 * x ** 2 - 4 * x

x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
plt.show();
'''

def init_param(args):
	parser = argparse.ArgumentParser(description="View binary file in plot with specific format, 1.0.0")
	parser.add_argument("-x", type=str, required=True, default="x.bin",
		help="input x binary filename")
	parser.add_argument("-y", type=str, required=True, default="y.bin",
		help="input y binary filename")
	parser.add_argument("-fx", type=str, required=False, default="float32",
		help="input x binary format: " + dtype_str)
	parser.add_argument("-fy", type=str, required=False, default="float32",
		help="input x binary format: " + dtype_str)
	parser.add_argument("-qx", type=int, required=False,
		help="Q value for x quantized data")
	parser.add_argument("-qy", type=int, required=False,
		help="Q value for y quantized data")

	parser.add_argument("-s", type=int, required=False,
		help="start. data = data[s:]")
	parser.add_argument("-e", type=int, required=False,
		help="end. data = data[:e]")

	parser.add_argument("-o", type=str, required=False, default="o.jpg",
		help="out image filename (jpg, png), do not imshow in live if gen output image")
	parser.add_argument("-v", action='store_true', required=False,
		help="verbose log like show data")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	bin_plot(args)
