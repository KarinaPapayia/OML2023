"""This file provides utility functions for the Exercise sheet #2"""

import numpy as np

def gen_sin_data(N=100, sigma=0.1, range_x=(0,1)):
    """Generate noisy sinusoidal 1D data"""
    x = np.random.uniform(*range_x, (N,))  # N points uniformly distributed on range_x
    epsilon = sigma*np.random.randn(N)  # outputs a normal distribution that we scale with sigma
    t = np.sin(2*np.pi*x) + epsilon  # takes sin and add noise
    return (x,t)

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
    # pos = np.concatenate((np.ones(1, N//2), pos), axis=0)  # add the bias term inside the points




