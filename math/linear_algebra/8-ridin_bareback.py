#!/usr/bin/env python3

"""matrix multiplication"""


def mat_mul(mat1, mat2):
    """matrix multiplication"""
    if len(mat1[0]) != len(mat2):
        return None

    result = []
    i = 0
    while i < len(mat1):
        row = []
        j = 0
        while j < len(mat2[0]):
            total = 0
            k = 0
            while k < len(mat2):
                total += mat1[i][k] * mat2[k][j]
                k += 1
            row.append(total)
            j += 1
        result.append(row)
        i += 1
    return result
