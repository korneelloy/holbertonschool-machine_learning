#!/usr/bin/env python3

"""stacked bar graph"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """fruits per person"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    categories = ["Farrah", "Fred", "Felicia"]

    plt.bar(categories, fruit[0], width=0.5, color="red", label="apples")
    plt.bar(
        categories,
        fruit[1],
        width=0.5,
        bottom=fruit[0],
        color="yellow",
        label="bananas",
    )
    plt.bar(
        categories,
        fruit[2],
        width=0.5,
        bottom=fruit[0] + fruit[1],
        color="#ff8000",
        label="oranges",
    )
    plt.bar(
        categories,
        fruit[3],
        width=0.5,
        bottom=fruit[0] + fruit[1] + fruit[2],
        color="#ffe5b4",
        label="peaches",
    )

    plt.legend(loc="upper right")

    plt.title("Number of Fruit per Person")
    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.yticks(range(0, 90, 10))

    plt.show()
