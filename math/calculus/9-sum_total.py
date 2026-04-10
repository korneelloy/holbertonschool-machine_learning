#!/usr/bin/env python3
"""sum of powers"""


def summation_i_squared(n):
    """return sum of squared number"""
    if not isinstance(n, (int)):
        return None
    return int(n * (n + 1) * (2 * n + 1) / 6)
