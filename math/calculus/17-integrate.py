#!/usr/bin/env python3


"""Calculation of the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Calculation of the integral of a polynomial"""
    if not isinstance(C, int) or not isinstance(poly, list) or len(poly) == 0:
        return None

    for x in poly:
        if not isinstance(x, (int, float)):
            return None

    new_list = [C]
    i = 0
    while i < len(poly):
        if poly[i] == 0:
            new_list.append(0)
        else:
            new_element = poly[i] / (i + 1)
            if new_element % 1 == 0:
                new_element = int(new_element)
            new_list.append(new_element)
        i += 1

    i = len(new_list) - 1
    while i > 0 and new_list[i] == 0:
        new_list.pop()
        i -= 1
    return new_list
