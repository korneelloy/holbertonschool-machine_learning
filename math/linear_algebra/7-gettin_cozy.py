#!/usr/bin/env python3


"""concatenate 2 matrices along axe"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenate 2 matrices along axis, default 0"""
    if len(mat1) != len(mat2) and len(mat1) != len(mat2[0]):
        return None
    if axis == 0:
        return mat1 + mat2
    new = []
    i = 0
    while i < len(mat1):
        line = mat1[i] + mat2[i]
        new.append(line)
        i += 1
    return new
