#!/usr/bin/env python3

"""moving everage"""

import numpy as np


def moving_average(data, beta):
    """
    calculate the weighted moving average of a data set:
    data is the list of data to calculate the moving average of
    beta is the weight used for the moving average
    Your moving average calculation should use bias correction
    Returns: a list containing the moving averages of data
    """
    list = []
    v = 0
    t = 0
    for x in data:
        t += 1
        v = beta * v + (1-beta) * x
        v_corrected = v / (1 - beta**t)
        list.append(v_corrected)
    return list
