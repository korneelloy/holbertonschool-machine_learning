#!/usr/bin/env python3


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices of the same size. Returns None if sizes differ."""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    new_mat = []
    i = 0
    while i < len(mat1):
        row = []
        j = 0
        while j < len(mat1[0]):
            row.append(mat1[i][j] + mat2[i][j])
            j += 1
        new_mat.append(row)
        i += 1

    return new_mat
