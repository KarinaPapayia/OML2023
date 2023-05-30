"""This file provides utility functions for the Homework #1"""

import numpy as np
from plot_utils import plot_SVM_sol


def gen_linsep_data(D=2, N=20, margin=1.):
    """Generate (linearbly separable) classification data in D dimension
    margin: the margin 

    Output: (x, c) where x is NxD and c is N"""
    w = np.random.uniform( -1, 1, (D,))  # random hyperplane orthogonal
    b = np.random.uniform(-1, 1)  # bias
    alpha = - b / np.linalg.norm(w)  # distance to the origin
    T = alpha * w / np.linalg.norm(w)  # translation to the hyperplane
    pts = 3*np.random.randn(N,D) + T.reshape(1, -1)


    cls = np.sign(pts @ w + b)  # compute the class
    pts = pts + cls[:, None]*  margin/2 * w[None, :] / (np.linalg.norm(w)**2)
    return (pts, cls)  # return the points and their classes




def augment(x):
    """Compute the augmented dataset
    x: (N,d) samples
    Return:
        X: (N, d+1) augmented samples (x 1)
    """
    N = x.shape[0]
    # create X = (x 1) in N x (D+1)
    X = np.concatenate((x, np.ones((N, 1))), axis=1)
    return X

if __name__ == "__main__":
    pts, cls = gen_spiral_data(N=200_000)
    plot_SVM_sol([0,1,0], pts, cls, 0)



