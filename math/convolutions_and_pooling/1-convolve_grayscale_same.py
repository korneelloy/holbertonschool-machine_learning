#!/usr/bin/env python3

"""same convolution on grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    same convolution on grayscale images:
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
    pad_h = (kH - 1) // 2
    pad_w = (kW - 1) // 2
    images_padded = np.pad(
        images,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant', constant_values=0
    )
    output = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(
                images_padded[:, i:i+kH, j:j+kW] * kernel, axis=(1, 2))
    return output
