#!/usr/bin/env python3

"""Pooling Forward Prop"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs forward propagation over a pooling layer of a neural network:

    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
    m is the number of examples
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    c_prev is the number of channels in the previous layer

    kernel_shape is a tuple of (kh, kw)
    containing the size of the kernel for the pooling
    kh is the kernel height
    kw is the kernel width

    stride is a tuple of (sh, sw) containing the strides for the pooling
    sh is the stride for the height
    sw is the stride for the width

    mode is a string containing either max or avg,
    indicating whether to perform maximum or average pooling, respectively

    Returns: the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    H_out = (h_prev - kh) // sh + 1
    W_out = (w_prev - kw) // sw + 1

    A = np.zeros((m, H_out, W_out, c_prev))
    for i in range(H_out):
        for j in range(W_out):
            region = A_prev[:, i*sh:i*sh+kh,
                            j*sw:j*sw+kw, :]
            if mode == 'max':
                A[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(region, axis=(1, 2))
    return A
