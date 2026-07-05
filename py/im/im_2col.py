
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data :由（数据量，通道，高，长）的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    # 获取 数据量、通道数、图像高度、图像长度
    N, C, H, W = input_data.shape
    # 对图像进行卷积运算后的输出高度，如图像是7X7，卷积核是5X5  结果是3X3
    out_h = (H + 2*pad - filter_h)//stride + 1
    # 对图像进行卷积运算后的输出宽度
    out_w = (W + 2*pad - filter_w)//stride + 1

	# 对图像在4个维度进行填充，默认pad为0
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    # 见下面解释1
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

   # 从左到右，从上到下依次进行遍历
    for y in range(filter_h):
        # 见解释2
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            # 见解释3
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

