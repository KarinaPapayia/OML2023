import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections

def plot_convergence(E, sg_norm, kmin):
    """Plot the convergence quantities (energy and gradient norm)
    E: (K,) energy 
    sg_norm: (K,) subgradient norm
    kmin: step at which the minimum loss is attained
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plt.suptitle("Convergence of the algorithm")
    axes[0].plot(E, label="$E(w^{(k)})$")
    axes[0].set_xlabel("$k$")
    axes[0].plot(kmin, E[kmin], c='r', marker='x', linestyle='none', label='$E_{\mathrm{min}}$')
    axes[0].legend()
    axes[1].plot(sg_norm, label="$\| \mathrm{sg}^{(k)} \|$")
    axes[1].set_xlabel("$k$")
    axes[1].plot(kmin, sg_norm[kmin], c='r', marker='x', linestyle='none', label='$\| \mathrm{sg}^{(k_\mathrm{min})} \|$')
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
    plt.plot(xx, -W[0]/W[1] * xx - W[2]/W[1], color="g", label="$\langle w^{(K)}, x\\rangle + b = 0$")
    plt.legend()
    plt.title(f"Solution at convergence, $K = {{}}$".format(cnt))
    plt.show()

def plot_SVM_dual_sol(alpha, x, Kfun, K, t, cnt):
    """Plot the SVM solution (2D)
    alpha: (n,) the dual solution 
    kfunc:  Kernel function
    t: (n,) targets in {-1, 1}
    cnt: number of iterations
    """
    fig, ax = plt.subplots()
    xplus = x[t == +1]
    xminus = x[t == -1]
    xx = np.linspace(x[:, 0].min(), x[:, 0].max())  # m new points
    yy = np.linspace(x[:, 1].min(), x[:, 1].max())  # m new points
    X, Y = np.meshgrid(xx, yy)
    pts = np.vstack((X.ravel(), Y.ravel())).T  # all the points on the grid , (Npoints,2)
    sprt_idx = np.where(alpha > 0)[0]  # indices for the SVM, only need to evaluate those
    Kpts = Kfun(pts, x[sprt_idx])   # of size (Npoints, n_sprt)

    y_w = (alpha[sprt_idx].reshape(1, -1) * t[sprt_idx].reshape(1, -1) * Kpts).sum(axis= 1)

    b = t[sprt_idx[0]] - (alpha[sprt_idx].reshape(1, -1) * t[sprt_idx].reshape(1, -1) * K[sprt_idx[0], sprt_idx]).sum(axis=1)  # compute the bias
    Z = (y_w + b).reshape(*X.shape)
    contf = ax.contourf(X, Y, Z, cmap='coolwarm')

    # support vectors
    xsprt = x[sprt_idx]
    tsprt = t[sprt_idx]
    xsprt_plus = xsprt[tsprt == +1]
    xsprt_minus = xsprt[tsprt == -1]

    # draw circles around support vectors
    coll_plus = matplotlib.collections.EllipseCollection(0.5*np.ones_like(xsprt_plus), 
                                                         0.5*np.ones_like(xsprt_plus),
                                                    np.zeros_like(xsprt_plus), offsets=xsprt_plus,
                                                    offset_transform=ax.transData, 
                                                edgecolor='blue', units='x', facecolor='none')

    coll_minus = matplotlib.collections.EllipseCollection(0.5*np.ones_like(xsprt_minus), 
                                                          0.5*np.ones_like(xsprt_minus),
                                                    np.zeros_like(xsprt_minus), offsets=xsprt_minus,
                                                    offset_transform=ax.transData, 
                                                edgecolor='orange', units='x', facecolor='none')
    ax.add_collection(coll_plus)
    ax.add_collection(coll_minus)

    # plot the training points
    ax.scatter(xplus[:, 0], xplus[:, 1], marker='+', label="$t_i = +1$")
    ax.scatter(xminus[:, 0], xminus[:, 1], marker='x', label="$t_i = -1$")
    cbar = plt.colorbar(contf)
    ax.legend()
    ax.set_title(f"Solution at convergence, $K = {{}}$".format(cnt))
    plt.show()
