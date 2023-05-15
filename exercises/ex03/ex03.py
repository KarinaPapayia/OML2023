import scipy as sp
import numpy as np
import utils
import matplotlib.pyplot as plt
import plot_utils as putils

def g(alpha, K, t):

    val = alpha.sum() - 1/2 * (alpha.reshape(-1, 1) * alpha.reshape(1, -1) * t.reshape(-1, 1) * t.reshape(1, -1) * K).sum()
    return val


def SMO(t, K, CNT_MAX=1000):
    """SMO algorithm

    t: (n,) targets
    K: (n,n) kernel 

    Ouput:
        alpha: (n,) dual solution
        """

    n = t.shape[0]  # number of samples
    alpha = np.zeros(n)  # initialize alpha
    not_KKT = where_not_KKT(alpha, t, K)  # samples violating the KKT conditions
    k = not_KKT[0] 
    cnt = 0 
    converged = False
    vals = [0.]
    while not converged:
        k = not_KKT[0]
        E_all = E(alpha, t, K)
        l = np.abs(E_all[k] - E_all).argmax()  # find the index l that maximizes the delta in E
        s = t[k] * t[l]  # 
        if s == 1:  # the bounds for the new alpha_l
            Lb =  0
            Hb = alpha[k] + alpha[l]
        else:
            Lb = max(0, alpha[l] - alpha[k])
            Hb = float('inf')
        alpha_max = alpha[l] + t[l] * (E_all[k] - E_all[l]) / (K[k, k] + K[l, l] - 2*K[k, l])
        alpha_l = np.clip(alpha_max, Lb, Hb)  # clip the value
        alpha_k = alpha[k] + t[k]*t[l]*(alpha[l] - alpha_l)
        alpha[l] = alpha_l
        alpha[k] = alpha_k
        # k = find_k(alpha, t, K)
        not_KKT = where_not_KKT(alpha, t, K)
        cnt += 1
        converged =  len(not_KKT) ==0 or cnt > CNT_MAX
        vals.append(g(alpha, K, t))
    return alpha, vals




def E(alpha, t, K):
    """Evaluate the bias-free energy f(x_i)-b - t_i
    alpha: (n,)
    t: (n,)
    K: (n,n)

    Output:
        E: (n,) the energy for each sample
    """
    y_w= (alpha.reshape(1,-1) * t.reshape(1, -1) * K).sum(axis=1) 
    return y_w - t

def where_not_KKT(alpha, t, K):
    """
    Compute where the KKT conditions are not satisfied
    The dual feasability is assumed always holding by design of the SMO algorithm
    Therefore, only primal feasability and complementary slackness are tested.
    """

    sprt_idx = np.where(alpha > 0)[0]  # indices for the support vectors
    if len(sprt_idx) == 0:  # e.g. at first pass
        return np.arange(len(alpha))  # return all the samples
    y_w = (alpha[sprt_idx].reshape(1, -1) * t[sprt_idx].reshape(1, -1) * K[:, sprt_idx]).sum(axis=1)
    b = t[sprt_idx[0]] - y_w[sprt_idx[0]]  # compute the bias

    not_primal_feas = (t * ( y_w + b) - 1 < 0)  # primal feasability

    not_comp_slack = np.abs(alpha * (t * ( y_w + b) - 1)) > 1e-8  # small epsilon

    return np.where(not_primal_feas | not_comp_slack)[0]


if __name__ == "__main__":

    X, t = utils.gen_spiral_data(N=100)


    def LinK(X, Y=None):
        if Y is None:
            Y = X
        return X @ Y.T 

    def ExpK(X, Y=None, sigma=2): 
        if Y is None:
            Y = X
        N, M = X.shape[0], Y.shape[0]
        return np.exp(-np.linalg.norm(X.reshape(N, 1, -1) - Y.reshape(1, M, -1), axis=-1)/(2*sigma**2))

    Kfun = ExpK
    K = Kfun(X)


    alpha,vals = SMO(t, K)
    cnt = len(vals)
    putils.plot_SVM_dual_sol(alpha, X, Kfun, K, t, cnt)
    plt.figure()
    plt.plot(alpha, linestyle='none', marker='.')

    plt.figure()
    plt.plot(vals, label='$g$')
    plt.legend()

    plt.show()

