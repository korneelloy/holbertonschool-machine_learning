#!/usr/bin/env python3


"""build descions tree"""


import numpy as np


class Node:
    """node in tree"""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """returns max depth below this node"""
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """counts nb of nodes below"""
        if only_leaves:
            return (
                self.left_child.count_nodes_below(only_leaves=only_leaves)
                + self.right_child.count_nodes_below(only_leaves=only_leaves))
        return (
            1
            + self.left_child.count_nodes_below(only_leaves=only_leaves)
            + self.right_child.count_nodes_below(only_leaves=only_leaves))

    def __str__(self):
        """Returns a string representation"""
        if self.is_root:
            node_str = (
                f"root [feature={self.feature},"
                f" threshold={self.threshold}]\n"
            )
        else:
            node_str = (
                f"-> node [feature={self.feature},"
                f" threshold={self.threshold}]\n"
            )
        left_text = str(self.left_child).rstrip("\n")
        right_text = str(self.right_child).rstrip("\n")
        left_str = self.left_child_add_prefix(left_text)
        right_str = self.right_child_add_prefix(right_text)
        return node_str + left_str + right_str

    def left_child_add_prefix(self, text):
        """add prefix to left child"""
        lines = text.split("\n")
        new_text = "    +--"+lines[0]+"\n"
        for x in lines[1:]:
            new_text += ("    |  "+x)+"\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """add prefix to right child"""
        lines = text.split("\n")
        new_text = "    +--"+lines[0]+"\n"
        for x in lines[1:]:
            new_text += ("       "+x)+"\n"
        return (new_text)

    def get_leaves_below(self):
        """returns leaves"""
        return (self.left_child.get_leaves_below()
                + self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """update bounds below"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

        self.left_child.lower[self.feature] = self.threshold
        self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """update indicator"""

        def is_large_enough(x):
            """returns a 1D numpy array of size
            `n_individuals` so that the `i`-th element of the later is `True`
            if the `i`-th individual has all its features > the lower bounds
            """

            return np.all(np.array([np.greater(x[:, key], self.lower[key])
                                    for key in self.lower.keys()]), axis=0)

        def is_small_enough(x):
            """
            returns a 1D numpy array of size
            `n_individuals` so that the `i`-th element of the later is `True`
            if the `i`-th individual has all its features <= the lower bounds
            """
            return np.all(np.array([np.less_equal(x[:, key], self.upper[key])
                                    for key in self.upper.keys()]), axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """predict"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """leaf in tree"""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """returns max depth below this leaf"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """counts nb of nodes below"""
        return 1

    def __str__(self):
        """Returns a string representation"""
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """retruns leaves"""
        return [self]

    def update_bounds_below(self):
        """update bounds below"""
        pass

    def pred(self, x):
        """predict"""
        return self.value


class Decision_Tree():
    """class tree itself"""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random",
                 root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """returens max depth of tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """counts nb of nodes below"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Returns a string representation"""
        return self.root.__str__()

    def get_leaves(self):
        """returns leaves"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """update bounds"""
        self.root.update_bounds_below()

    def pred(self, x):
        """predict"""
        return self.root.pred(x)

    def update_predict(self):
        """predict"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.sum(
            [leaf.indicator(A) * leaf.value for leaf in leaves],
            axis=0)
