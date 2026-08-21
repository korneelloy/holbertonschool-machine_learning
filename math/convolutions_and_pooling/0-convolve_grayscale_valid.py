#!/usr/bin/env python3

import numpy as np

"""convolution on grayscale images"""


def convolve_grayscale_valid(images, kernel):
    """
    convolution on grayscale images:
    images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
    kh is the height of the kernel
    kw is the width of the kernel
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kH, kW = kernel.shape
    out_H = h - kH + 1
    out_W = w - kW + 1
    output = np.zeros((m, out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            output[:, i, j] = np.sum(
                images[:, i:i+kH, j:j+kW] * kernel, axis=(1, 2))
    return output
