import utils
import plot_utils
import numpy as np
from collections import defaultdict
import scipy as sp



def pred(X, W):
    """
    X (n, d+1)
    W (d+1, )
    """
    return X@W

def Loss(X, W, t, p=10):
    # return 1/2*sp.linalg.norm(W[:-1])**2 +  1/p * np.log(1 + np.exp(p*(1 - pred(X, W) * t)))
    return   1/p * np.log(1 + np.exp(p*(1 - pred(X, W) * t)))

def Grad(X, W, t, p=10):
    E = np.exp(-p*(1 - pred(X, W) * t))
    # return np.concatenate((W[:-1], np.zeros(1)), axis=0)  + np.mean(- X * t[:, None] /(1 + E[:, None]), axis=0)
    return   np.mean(- X * t[:, None] /(1 + E[:, None]), axis=0)

def SGD(x, t, bs=1, lr=0.1, EPOCHS=5):
    """
    Batch SGD implementation

    x: (N, d) samples 
    t: (N,) targets 
    bs: batch size 
    lr: learning rate
    EPOCHS: number of epochs

    Return:
        qts: training quantities
        """

    n = x.shape[0]  # total number of samples
    d = x.shape[1]
    W = np.zeros(d+1)
    nepoch =0
    converged = False
    batch_id = 0
    qts = defaultdict(list)
    Emin = float('inf')
    niter = 0
    cnt = 0

    # we implement the batched SGD (without replacement):
    # we draw a random permutation at each epoch start and sample the batches
    # according to it, until the epoch is over

    while not converged:
        if batch_id*bs >= n:  # we reached the end of an epoch
            nepoch += 1
            batch_id = 0  # reset batch indx
        if batch_id == 0:  # start a new epoch
            perm = np.random.permutation(n)  # new permutation
        
        batch_idx = perm[batch_id*bs:min(n, (batch_id+1)*bs)]
        batch = utils.augment(x[batch_idx, :])
        # B = len(batch_idx)
        grad = Grad(batch, W, t[batch_idx])
        W =W  - lr * Grad(batch, W, t[batch_idx])
        converged = nepoch > EPOCHS
        batch_id += 1  # end of a batch
        E = Loss(batch, W, t[batch_idx]).mean()
        cnt += 1
        if E < Emin:
            Emin = E
            kmin = cnt
        qts['E'].append(E)
        qts['g_norm'].append(sp.linalg.norm(grad))
        niter += 1
    qts['niter'] = niter
    qts['W'] = W
    qts['kmin'] = kmin
    qts['Emin'] = Emin
        # qts['loss'].append(Loss(X, W, t).mean())
    return qts

if __name__ == "__main__":
    pts, cls = utils.gen_linsep_data(N=100_000, margin=1)

    qts = SGD(pts, cls, bs=100, lr=0.01, EPOCHS=3)
    W = qts['W']
    niter = qts['niter']
    kmin = qts['kmin']
    plot_utils.plot_SVM_sol(W, pts, cls, kmin)
    plot_utils.plot_convergence(qts['E'], qts['g_norm'])




