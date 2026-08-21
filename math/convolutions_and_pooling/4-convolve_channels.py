#!/usr/bin/env python3

"""convolution with padding on color images"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Write a function that performs a convolution on color images:

    images is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for
    the convolution
    kh is the height of the kernel
    kw is the width of the kernel
    padding is either a tuple of (ph, pw), 'same', or 'valid'
    if 'same', performs a same convolution
    if 'valid', performs a valid convolution
    if a tuple:
    ph is the padding for the height of the image
    pw is the padding for the width of the image
    the image should be padded with 0's
    stride is a tuple of (sh, sw)
    sh is the stride for the height of the image
    sw is the stride for the width of the image
    You are only allowed to use two for loops; any other loops of any kind are
    not allowed Hint: loop over i and j
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kH, kW, c = kernel.shape
    pad_h = 0
    pad_w = 0
    if (padding == 'same'):
        pad_h = (kH - 1) // 2
        pad_w = (kW - 1) // 2
    elif (padding != 'valid'):
        pad_h = padding[0]
        pad_w = padding[1]
    images_padded = np.pad(
        images,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant', constant_values=0
    )
    sh = stride[0]
    sw = stride[1]
    out_H = (h + 2 * pad_h - kH) // sh + 1
    out_W = (w + 2 * pad_w - kW) // sw + 1

    output = np.zeros((m, out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            output[:, i, j] = np.sum(
                images_padded[:, i*sh:i*sh+kH, j*sw:j*sw+kW] * kernel,
                axis=(1, 2, 3))
    return output
