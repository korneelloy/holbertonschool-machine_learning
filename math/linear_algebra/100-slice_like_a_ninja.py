#!/usr/bin/env python3


"""
Write a function  that slices a matrix along specific axes:

You can assume that matrix is a numpy.ndarray
You must return a new numpy.ndarray
axes is a dictionary where the key is an axis to slice
along and the value is a tuple representing the slice to make along that axis
You can assume that axes represents a valid slice

"""


def np_slice(matrix, axes={}):
    slices = [slice(None)] * matrix.ndim
    print("slices", slices)
    for axis, value in axes.items():
        slices[axis] = slice(*value)
    return matrix[tuple(slices)]
