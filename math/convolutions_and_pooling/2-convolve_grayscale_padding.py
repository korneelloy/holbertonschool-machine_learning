#!/usr/bin/env python3

"""convolution with padding on grayscale images"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    convolution with padding on grayscale images:
    images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
    kh is the height of the kernel
    kw is the width of the kernel
    padding is a tuple of (ph, pw)

    ph is the padding for the height of the image
    pw is the padding for the width of the image

    the image should be padded with 0's
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kH, kW = kernel.shape
    pad_h = padding[0]
    pad_w = padding[1]
    images_padded = np.pad(
        images,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant', constant_values=0
    )
    out_H = h + 2 * pad_h - kH + 1
    out_W = w + 2 * pad_w - kW + 1
    output = np.zeros((m, out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            output[:, i, j] = np.sum(
                images_padded[:, i:i+kH, j:j+kW] * kernel, axis=(1, 2))
    return output
