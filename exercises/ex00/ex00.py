# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.0
#   kernelspec:
#     display_name: OML23
#     language: python
#     name: oml23
# ---

# # Basic Numpy / Scipy manipulations
#
# ## Intro
#
# This notebook will guide you with some basic numerical manipulations in
# Python, with the Numpy and Scipy packages.
#
#
# **It assumes you have read and followed the instructions of `instructions.md`
# file (in the same directory)** in order to setup the virtual environment
# (best viewed on
# [GitHub](https://github.com/KarinaPapayia/OML2023/blob/main/exercises/ex00/instructions.md)).
#
#
# The kernel you see on top should be the one you have installed with the
# command `ipython kernel install`. **If it's not the case, select it with
# Kernel > Change Kernel > OML23.**
#
# **Restart the Kernel might sometime help if you get erros or freezes**
#
# If this is the first Jupyter notebook you see: you run the cells with the
# commands on the top or with `Shift-Enter`. 
#
# General Python refresher: [Learn Python in Y
# minutes](https://learnxinyminutes.com/docs/python/).
#
# *Please send an email to brechet (at) mpi (dot) mis (dot) de if you have a
# question / if something is not working as planned!* 

# --- 

import numpy as np

# ## Arrays in numpy

# Numpy can manage $n$-dimensional arrays with the type `numpy.ndarray`. A 1-d
# array (vector) can be created from a (Python) list like so:

a = np.array([1, 2, 3])
print('shape of a:', a.shape)

# Arrays can also be created with built-in functions of Numpy (see [Numpy
# documentation](https://numpy.org/doc/1.24/user/index.html#user) for more info and examples).

b = np.arange(1, 10) # (Integer) interval [begin, end)
c = np.ones((3, 2))  # 1s, shape given
d = np.zeros_like(c)  # also works with ones_like, empty_like
e = np.empty((2, 5))  # has to be initialized
print('e:', e)

f = np.linspace(0, 1, num=100)  # linear space

# Numpy has broadcasting abilities: two arrays can be added, multiplied, etc.
# together if their shapes are compatible, i.e., if they are the same or if one
# of the array has dimension 1. 

# For instance, the two arrays `a` and `b` are not compatible as it is, since
# their shapes are: 

print('shape of a:', a.shape)
print('shape of b:', b.shape)

# so in order to make them compatible we use the `reshape` method a put a `1`
# where we want the array to be copied, and `-1` to infer the missing dimension
# (internally computed).

print('shape of a.reshape(-1, 1):', a.reshape(-1, 1).shape)


# to create a third array with entries from `a` and `b`, we then simply use a
# vectorized computation like

c = a.reshape(-1, 1) * b.reshape(1, -1)  # see a as a column and b as a row

# It creates an array $c$ with entries $c_{ij} = a_i b_j$.

print('c:', c)


# ### Basic function vectorization 

# **Perform as little loops as possible!**

# Ex: Find $S_{100} := \sum_{i=1}^{100} \frac{1}{i^2}$ 

# Build up the function from the built-in

print("S_100 = ", ((1/np.arange(1, 101))**2).sum())

# 2D functions

import matplotlib.pyplot as plt

t = np.linspace(0, 0.01, num=1000)  # 1/100 second
f1 = 440
f2 = 220
tt1 = 2*np.pi*f1*t  # vector like t
tt2 = 2*np.pi*f2*t

v1 = np.sin(tt1)  # sinus applied to tt1
v2 = np.sin(tt2)  # sinus applied to tt2
v3 = v1 + v2    

plt.figure()
# Ploting
plt.plot(t, v1, label=f"{f1} Hz")
plt.plot(t, v2, label=f"{f2} Hz")
plt.plot(t, v3, label=f"v3")
plt.xlabel('time')
plt.ylabel('y')
plt.legend()
plt.show()

# Gaussian in 2d
x = np.linspace(-2, 2, num=100).reshape(1, -1)  # will be broadcasted along dim 0
y = np.linspace(-2, 2, num=100).reshape(-1, 1)  # will be broadcasted along dim 1


sigma_x =1
sigma_y = 0.5


Z = np.exp(-1/2 * ((x/sigma_x)**2 + (y/sigma_y)**2))  # 2D Gaussian pdf

X, Y = np.meshgrid(x, y)  # makes 2D matrices out of 1D vectors

plt.figure()
plt.contour(X, Y, Z) # show the contour of the function in Z on the mesh X, Y
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# ## Simple image manipulation

import scipy as sp
from skimage import data # for images

# Load an image with scikit (by default between 0 and 255, make the range in [0,1]):

camera = data.camera()/255

print('shape of camera:', camera.shape)

# Black and white image, $C = {(C_{ij})}_{i,j=1}^{n, m}$. Each $C_{ij} \in
# [0,1]$.

camera[0,0], camera[511, 511]

# Perform a SVD to find orthogonal components in the image: $A = USV^\top$ with orthonormal $U$ and $V$, and diagonal $S = \rm{Diag}(\sigma_1, \ldots, \sigma_k)$. 


U, S, Vt = sp.linalg.svd(camera)  # performs a SVD, with decreasing singular
                                    #   values

# `S` is a vector, the original image is found with `U @ (S.reshape(-1,1) *
# Vt)` :

print(np.allclose(camera, U @ (S.reshape(-1, 1) * Vt)))

# `@` is the  matrix multiplication, and the vector `S` is reshaped so that
# each of its entry multiplies a different row in `Vt`. (same effect as when
# $S$ is a diagonal matrix and $V^\top$: the rows of $V^\top$ are also
# multiplied by the entries in $S$).


# We can use the SVD to perform (approximate) compression of the data, by
# selection a given number of singular values in $S$ and not all. 



N = S.shape[0]  # total number of singular values, same as width / height


print("Total number of components: ", N)
p = 1  # number of components we keep
C2 = U[:, :p] @ (S[:p].reshape(-1, 1) * Vt[:p, :])  # select the p first
                                                    # components from the SVD


fig, axes = plt.subplots(1, 2)
axes[0].imshow(camera, cmap='gray', vmin=0, vmax=1)  # gives the range of inputs [0,1]
axes[1].imshow(C2, cmap='gray', vmin=0, vmax=1)

# The singular values encode the variation in the image, since if $A =
# USV^\top = U \rm{Diag}(\sigma_1, \ldots, \sigma_k) V^\top$ is a SVD of a
# matrix $A$, then $\|A\|_F = \sqrt{\sum_i \sigma_i^2}$.

# Define the function computing the norm from the vector of singular values

"""Compute the norm based on the singular values S (a vector)"""
def norm_F(S): return np.sqrt((S**2).sum())  # 

# Check if the formula is correct, compute $\|A\|_F$ with `np.linalg.norm(A, 'fro')`:

import math


print(math.isclose(norm_F(S), np.linalg.norm(camera, 'fro')))    # should be 0

# We therefore define the *reconstruction* to the ratio between the norm of the
# compressed image to the original one as below. 

# Define the function for the reconstruction rate as


def reconstruction_rate(p, S=S):  # default value to the already defined S
    """Compute the reconstruction rate given a number of components"""
    norm_orig = norm_F(S)
    return 100* (norm_F(S[:p]) / norm_orig)

# Define the compression factor as the ratio between the total number of
# components and the number kept

def compression_factor(p, N=N): return (N/p)


print("Reconstruction rate: {:.3f}%".format(reconstruction_rate(p)))
print("Compression factor {:.0f}x".format(compression_factor(p)))
plt.show()

# **Question** : how many components do we need in order to get at least 99% reconstruction
# rate with the highest compression factor?

# (Hint: use a loop!)

# +
# enter your answer here...


# -

# **Answer**

# We simply loop over `p` with a condition on the reconstrction rate

p = 1
rrate = reconstruction_rate(p)
while rrate < 99:
    p += 1
    rrate = reconstruction_rate(p)

print(f"Number of components: {p}") # f-strings allow to format strings in
                                    # Python with variables

print("Reconstruction rate: {:.3f}%".format(reconstruction_rate(p)))
print("Compression factor: {:.0f}x".format(compression_factor(p)))

C3 = U[:, :p] @ (S[:p].reshape(-1, 1) * Vt[:p, :])  # select the p first
                                                    # components from the SVD


fig, axes = plt.subplots(1, 3)
axes[0].imshow(camera, cmap='gray', vmin=0, vmax=1)
axes[0].set_title('Original')
axes[1].imshow(C2, cmap='gray', vmin=0, vmax=1)
axes[1].set_title('p = 1')
axes[2].imshow(C3, cmap='gray', vmin=0, vmax=1)
axes[2].set_title(f'p = {p}')
plt.show()
