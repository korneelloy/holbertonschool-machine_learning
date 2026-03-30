#!/usr/bin/env python3

"""returns shape of matrix"""


def matrix_shape(matrix):
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else []
    return shape
