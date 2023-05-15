"""This file provides utility functions for the Homework #1"""

import numpy as np


def gen_linsep_data(D=2, N=20):
    """Generate (linearbly separable) classification data in D dimension

    Output: (x, c) where x is NxD and c is N"""

    w = np.random.uniform( -1, 1, (D,))  # random hyperplane orthogonal
    b = np.random.uniform(-1, 1)  # bias
    alpha = - b / np.linalg.norm(w)  # distance to the origin
    T = alpha * w / np.linalg.norm(w)  # translation to the hyperplane
    pts = np.random.uniform(-1,1, (N,D)) + T.reshape(1, -1)
    cls = np.sign(pts @ w + b)  # compute the class
    return (pts, cls)  # return the points and their classes




def gen_spiral_data(N=20):
    """Generate (linearbly separable) classification data in D dimension

    Output: (x, c) where x is NxD and c is N"""

    # N = 50  # number of points per class
    theta = np.linspace(0, 2*np.pi, N)  # angles
    R = np.linspace(1, 5, N).reshape(-1, 1)  # radii
    centers_plus = R * np.vstack(( np.cos(theta), np.sin(theta) )).T
    centers_minus = R * np.vstack(( np.cos(theta+np.pi), np.sin(theta+np.pi) )).T

    xplus = centers_plus + np.random.rand(N, 2)
    xminus = centers_minus + np.random.rand(N, 2)

    pts = np.vstack((xplus, xminus))
    t = np.vstack((np.ones(N), -np.ones(N))).ravel()
    rp = np.random.permutation(2*N)
    pts = pts[rp, :]
    cls = t[rp]
    return (pts, cls)  # return the points and their classes
