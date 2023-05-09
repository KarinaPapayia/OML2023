"""This file provides utility functions for the Homework #1"""

import numpy as np


def gen_binary_data(D=2, N=20):
    """Generate (linearbly separable) classification data in D dimension

    Output: (x, c) where x is NxD and c is N"""

    w = np.random.uniform( -1, 1, (D,))  # random hyperplane orthogonal
    b = np.random.uniform(-1, 1)  # bias
    alpha = - b / np.linalg.norm(w)  # distance to the origin
    T = alpha * w / np.linalg.norm(w)  # translation to the hyperplane
    pts = np.random.uniform(-1,1, (N,D)) + T.reshape(1, -1)
    cls = np.sign(pts @ w + b)  # compute the class
    return (pts, cls)  # return the points and their classes




