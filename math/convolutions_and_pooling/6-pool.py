#!/usr/bin/env python3

"""convolution with padding on color images"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    performs pooling on images:
    images is a numpy.ndarray with shape (m, h, w, c)
    containing multiple images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    c is the number of channels in the image

    kernel_shape is a tuple of (kh, kw) containing the kernel shape
    kh is the height of the kernel
    kw is the width of the kernel

    stride is a tuple of (sh, sw)
    sh is the stride for the height of the image
    sw is the stride for the width of the image

    mode indicates the type of pooling
    max indicates max pooling
    avg indicates average pooling
    You are only allowed to use two for loops;
    Returns: a numpy.ndarray containing the pooled images
    """

    m, h, w, c = images.shape
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]

    out_H = (h - kh) // sh + 1
    out_W = (w - kw) // sw + 1
    output = np.zeros((m, out_H, out_W, c))

    for i in range(out_H):
        for j in range(out_W):
            region = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            if (mode == 'max'):
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            elif (mode == 'avg'):
                output[:, i, j, :] = np.mean(region, axis=(1, 2))

    return output
