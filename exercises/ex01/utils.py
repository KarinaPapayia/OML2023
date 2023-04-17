"""Some utilities functions"""

import matplotlib.pyplot as plt
from matplotlib import cm  # colormaps
 
def plot_GD(grid, GDout, eta):
    """Plot the results of gradient descent vs the original loss
    grid = (W1, W2, Z): the grid onto which we plot
    GDout = (Warr, EW): the results of GD. Warr is a 2D array Tx2, EW is the value of
    the loss at those locations"""

    W1, W2, Z = grid
    Warr, EW = GDout
    T = Warr.shape[0] -1  # total number of points
    fig = plt.figure(figsize=(14, 4))  # to plot the training 
# axes[0].contour(W1, W2, Z, cmap=cm.coolwarm, antialiased=True, label="E") # the function on the grid
# axes[0].scatter(Warr[:, 0], Warr[:, 1], color='orange')  # convert the list of 2D

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')  # parameters are N_row, N_col, N_row
    surf = ax1.plot_surface(W1, W2, Z, cmap=cm.coolwarm, antialiased=True,
                            alpha=0.5, zorder=1)
    ax1.set_xlabel("$w_1$")
    ax1.set_ylabel("$w_2$")
    ax1.set_zlabel("$E(w)$")
    ax1.set_title("3D plot")

    ax1.scatter(Warr[:,0], Warr[:, 1], EW, color='orange', zorder=4,
                label='$w{(k)}$')  # add the points
# Contour plot
    ax2 = fig.add_subplot(1, 3, 2)  # parameters are N_row, N_col, index

# (Filled) Contour plot
    cont = ax2.contourf(W1, W2, Z, cmap=cm.coolwarm)
    ax2.set_xlabel("$w_1$")
    ax2.set_ylabel("$w_2$")
    ax2.set_title("Contour plot")

    ax2.scatter(Warr[:, 0], Warr[:, 1], color='orange')
    plt.text(Warr[0,0] + 0.05, Warr[0,1] + 0.05, "$w^{(0)}$", fontsize=12, color="orange")
    plt.text(Warr[-1,0] + 0.05 , Warr[-1,1] + 0.05, "$w^{(T)}$", fontsize=12, color="orange")

# plt.tight_layout()  # better margins
#color bar
# fig.subplots_adjust(right=0.8)
# cbar_ax=fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(cont, cax=cbar_ax)
# axes[0].set_title("")
                                                           # points to a matrix

    ax3 = fig.add_subplot(1, 3, 3)


    ax3.plot(EW, marker='o', color='orange', linestyle='none') 
    ax3.set_title("$E$ during training")
    ax3.set_xlabel("$k$")
    ax3.set_ylabel("$E$")

    fig.suptitle(r"Gradient descent with $\eta={}$, $T={}$".format(eta, T))
    plt.show()
