#!/usr/bin/env python3

"""Pooling Back prop"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs back propagation over a pooling layer of a neural network:

    dA is a numpy.ndarray of shape (m, h_new, w_new, c_new)
    containing the partial derivatives with respect to the output of the
    pooling layer
    m is the number of examples
    h_new is the height of the output
    w_new is the width of the output
    c is the number of channels

    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c)
    containing the output of the previous layer
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer

    kernel_shape is a tuple of (kh, kw) containing the
    size of the kernel for the pooling
    kh is the kernel height
    kw is the kernel width

    stride is a tuple of (sh, sw) containing the strides for the pooling
    sh is the stride for the height
    sw is the stride for the width

    mode is a string containing either max or avg,
    indicating whether to perform maximum or average pooling, respectively

    Returns:partial derivatives with respect to the previous layer (dA_prev)
    """

    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            if mode == 'max':
                region = A_prev[:, vert_start:vert_end,
                                horiz_start:horiz_end, :]
                max_val = np.max(region, axis=(1, 2), keepdims=True)
                mask = (region == max_val)
                dA_prev[:, vert_start:vert_end,
                        horiz_start:horiz_end, :] += (
                    mask * dA[:, i:i+1, j:j+1, :])
            elif mode == 'avg':
                dA_prev[:, vert_start:vert_end,
                        horiz_start:horiz_end, :] += (
                    dA[:, i:i+1, j:j+1, :] / (kh * kw))
    return dA_prev
