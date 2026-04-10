#!/usr/bin/env python3


def poly_derivative(poly):
    """function returning derivative"""
    if not isinstance(poly, list):
        return None
    if len(poly) == 1:
        return [0]
    i = 1
    derivative = []
    while i < len(poly):
        if not isinstance(poly[i], int):
            return None
        derivative.append(poly[i] * i)
        i += 1
    return derivative
