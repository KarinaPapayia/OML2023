import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(E, sg_norm):
    """Plot the convergence quantities (energy and gradient norm)
    E: (K,) energy 
    sg_norm: (K,) subgradient norm
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plt.suptitle("Convergence of the algorithm")
    axes[0].plot(E, label="$E(w^{(k)})$")
    axes[0].set_xlabel("$k$")
    axes[0].legend()
    axes[1].plot(sg_norm, label="$\| \mathrm{sg}^{(k)} \|$")
    axes[1].set_xlabel("$k$")
    axes[1].legend()
    plt.show()
    return fig, axes

def plot_SVM_sol(W, x, t, cnt):
    """Plot the SVM solution (2D)
    W: (3,) the solution hyperplane
    x: (N, 2) 2D samples
    t: (N,) targets in {-1, 1}
    cnt: number of iterations
    """
    plt.figure()
    xplus = x[t == +1]
    xminus = x[t == -1]
    plt.scatter(xplus[:, 0], xplus[:, 1], marker='+', label="$t_i = +1$")
    plt.scatter(xminus[:, 0], xminus[:, 1], marker='x', label="$t_i = -1$")
    xx = np.linspace(x[:, 0].min(), x[:, 0].max())
    plt.plot(xx, -W[0]/W[1] * xx - W[2]/W[1], color="g", label="$\langle w^{(K)},
             x\\rangle + b = 0$")
    plt.legend()
    plt.title(f"Solution at convergence, $K = {{}}$".format(cnt))
    plt.show()
