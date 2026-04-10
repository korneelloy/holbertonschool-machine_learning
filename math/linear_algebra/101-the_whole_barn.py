#!/usr/bin/env python3


"""

Write a function  that adds two matrices:


If matrices are not the same shape, return None
You can assume that mat1 and mat2 will never be empty
"""


def add_matrices(mat1, mat2):

    if not isinstance(mat1, list):
        return mat1 + mat2

    if len(mat1) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        sub = add_matrices(mat1[i], mat2[i])
        if sub is None:
            return None
        result.append(sub)

    return result
