#!/usr/bin/env python3

"""plotting Y solid red / x range 0-10"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """plotting Y solid red / x range 0-10"""
    x = np.arange(0, 11)
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.xlim(0, 10)
    plt.plot(x, y, "r")
    plt.show()
