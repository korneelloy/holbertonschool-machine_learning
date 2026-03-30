#!/usr/bin/env python3


"""module for flipping matrices over their diagonal"""


def matrix_transpose(matrix):
    """Returns the transposed version of a 2D matrix."""
    reversed_matrix = []
    i = 0
    while i < len(matrix[0]):
        row = []
        j = 0
        while j < len(matrix):
            row.append(matrix[j][i])
            j += 1
        reversed_matrix.append(row)
        i += 1

    return reversed_matrix
