import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'
plt.rc('text', usetex=True)
import sinkhorn_pointcloud as spc
import utils
import torch
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'
plt.rc('text', usetex=True)


def plot_distances(dimensions, sigmas2):
    cmap = matplotlib.cm.get_cmap('Set1')  # color map
    # Load results
    with open(os.path.join(dirname, "W2_empirical"), "rb") as f:
        W2_empirical = pickle.load(f)
    with open(os.path.join(dirname, "SW2"), "rb") as f:
        SW2 = pickle.load(f)
    with open(os.path.join(dirname, "Sinkhorn"), "rb") as f:
        Sinkhorn = pickle.load(f)
    with open(os.path.join(dirname, "SlicedSinkhorn"), "rb") as f:
        SlicedSinkhorn = pickle.load(f)
    with open(os.path.join(dirname, "MMD2"), "rb") as f:
        MMD2 = pickle.load(f)
    with open(os.path.join(dirname, "SlicedMMD2"), "rb") as f:
        SlicedMMD2 = pickle.load(f)

    # Plot divergences against parameter sigma^2
    N_d = len(dimensions)
    if N_d == 1:
        fig, axs = plt.subplots(1, N_d, figsize=(4.5, 3.5))
    else:
        fig, axs = plt.subplots(1, N_d, figsize=(8, 3))
    for i_d in range(N_d):
        d = dimensions[i_d]
        if N_d == 1:
            ax = axs
        else:
            ax = axs[i_d]

        ax.plot(sigmas2, W2_empirical[:, i_d, :].mean(axis=0), color=cmap(0), marker=".")
        ax.plot(sigmas2, SW2[:, i_d, :].mean(axis=0), color=cmap(0), marker=".", ls='--')
        ax.plot(sigmas2, Sinkhorn[:, i_d, :].mean(axis=0), color=cmap(1), marker=".")
        ax.plot(sigmas2, SlicedSinkhorn[:, i_d, :].mean(axis=0), color=cmap(1), ls='--', marker=".")
        ax.plot(sigmas2, MMD2[:, i_d, :].mean(axis=0) ** (1 / 2), color=cmap(2), marker=".")
        ax.plot(sigmas2, SlicedMMD2[:, i_d, :].mean(axis=0), color=cmap(2), marker=".", ls='--')

        # Plot error bands
        # ax.fill_between(sigmas2, np.percentile(W2_empirical[:, i_d, :], 10, axis=0), np.percentile(W2_empirical[:, i_d, :], 90, axis=0), facecolor=cmap(0), alpha=0.2)
        # ax.fill_between(sigmas2, np.percentile(SW2[:, i_d, :], 10, axis=0), np.percentile(SW2[:, i_d, :], 90, axis=0), facecolor=cmap(0), alpha=0.2)
        # ax.fill_between(sigmas2, np.percentile(Sinkhorn[:, i_d, :], 10, axis=0), np.percentile(Sinkhorn[:, i_d, :], 90, axis=0), facecolor=cmap(1), alpha=0.2)
        # ax.fill_between(sigmas2, np.percentile(SlicedSinkhorn[:, i_d, :], 10, axis=0), np.percentile(SlicedSinkhorn[:, i_d, :], 90, axis=0), facecolor=cmap(1), alpha=0.2)
        # ax.fill_between(sigmas2, np.percentile(MMD2[:, i_d, :]**(1/2), 10, axis=0), np.percentile(MMD2[:, i_d, :]**(1/2), 90, axis=0), facecolor=cmap(2), alpha=0.2)
        # ax.fill_between(sigmas2, np.percentile(SlicedMMD2[:, i_d, :], 10, axis=0), np.percentile(SlicedMMD2[:, i_d, :], 90, axis=0), facecolor=cmap(2), alpha=0.2)

        ax.set_xlabel(r'$\sigma^2$', fontsize=14)
        ax.set_xticks(np.arange(0., 9. + 1, 2.0))
        ax.set_ylabel("divergence", fontsize=12)
        # ax.set_title(r'$d =\ $' + str(d), fontsize=16)
    # Add legend
    if N_d == 1:
        lgd = fig.legend(labels=['Wasserstein', 'Sliced-Wasserstein', 'Sinkhorn', 'Sliced-Sinkhorn', 'MMD', 'Sliced-MMD'],
                         loc='lower left', bbox_to_anchor=(0.04, 0.96, 1, 0.2), borderaxespad=0, handletextpad=0.1,
                         columnspacing=0.2, ncol=3, fancybox=False, fontsize=11, frameon=False)
    else:
        lgd = fig.legend(labels=['Wasserstein', 'Sliced-Wasserstein', 'Sinkhorn', 'Sliced-Sinkhorn', 'MMD', 'Sliced-MMD'],
                         loc='upper center', bbox_to_anchor=[0.485, 0.54], borderaxespad=9, handletextpad=0.1, columnspacing=0.2, ncol=8,
                         fancybox=False, fontsize=11, frameon=False)
    axins = ax.inset_axes([0.12, 0.4, 0.9, 0.26])
    axins.plot(sigmas2, SW2[:, i_d, :].mean(axis=0), color=cmap(0), marker=".", ls='--')
    axins.plot(sigmas2, SlicedSinkhorn[:, i_d, :].mean(axis=0), color=cmap(1), marker=".", ls='--')
    axins.plot(sigmas2, MMD2[:, i_d, :].mean(axis=0)**(1/2), color=cmap(2), marker=".")
    axins.plot(sigmas2, SlicedMMD2[:, i_d, :].mean(axis=0), color=cmap(2), marker=".", ls='--')

    # axins.fill_between(sigmas2, np.percentile(SW2[:, i_d, :], 10, axis=0), np.percentile(SW2[:, i_d, :], 90, axis=0),
    #                 facecolor=cmap(0), alpha=0.2)
    # axins.fill_between(sigmas2, np.percentile(SlicedSinkhorn[:, i_d, :], 10, axis=0),
    #                 np.percentile(SlicedSinkhorn[:, i_d, :], 90, axis=0), facecolor=cmap(1), alpha=0.2)
    # axins.fill_between(sigmas2, np.percentile(MMD2[:, i_d, :] ** (1 / 2), 10, axis=0),
    #                 np.percentile(MMD2[:, i_d, :] ** (1 / 2), 90, axis=0), facecolor=cmap(2), alpha=0.2)
    # axins.fill_between(sigmas2, np.percentile(SlicedMMD2[:, i_d, :], 10, axis=0),
    #                 np.percentile(SlicedMMD2[:, i_d, :], 90, axis=0), facecolor=cmap(2), alpha=0.2)

    x1, x2, y1, y2 = -0.1, 9.2, -0.05, 0.6
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    ax.indicate_inset_zoom(axins)
    fig.tight_layout()
    fig.savefig(os.path.join(dirname, "results.pdf"), bbox_inches='tight', bbox_extra_artists=(lgd,))
    plt.close(fig)


if __name__ == '__main__':
    compute_dist = True
    n_runs = 10
    dims = [10]
    N_d = len(dims)

    # Data-generating parameters
    sigma2_star = 4.
    mean_star = 0.
    N_X = 1000  # number of observations

    # Hyperparameters for Sinkhorn divergences
    eps = 1  # regularization
    niter = 10000  # maximum number of iterations

    # Create directory that will contain the results
    dirname = os.path.join("results", "motivation_" + str(N_X) + "obs_dim=" + str(dims[0]) + "to" + str(dims[-1]))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Parameters tested
    T = 20
    sigmas2 = np.linspace(0.1, 9.0, num=T)
    sigmas2 = np.concatenate((sigmas2, [sigma2_star]))  # We add the true parameter
    sigmas2 = np.sort(sigmas2)

    if compute_dist:
        W2_empirical = np.zeros((n_runs, N_d, T+1))
        SW2 = np.zeros((n_runs, N_d, T+1))
        Sinkhorn = np.zeros((n_runs, N_d, T + 1))
        SlicedSinkhorn = np.zeros((n_runs, N_d, T + 1))
        MMD2 = np.zeros((n_runs, N_d, T + 1))
        SlicedMMD2 = np.zeros((n_runs, N_d, T + 1))
        n_proj = 50
        for nr in range(n_runs):
            print("Run " + str(int(nr)))
            for i_d in range(N_d):
                # Generate observations from the target Gaussian (with parameter sigma2_star)
                d = dims[i_d]
                X = np.random.multivariate_normal(mean_star * np.ones(d), sigma2_star * np.eye(d), N_X)  # observations
                for k in range(T+1):
                    print("sigma:" + str(sigmas2[k]))
                    # Generate data from Gaussian distribution with parameter sigma2[k]
                    Y = np.random.multivariate_normal(mean_star * np.ones(d), sigmas2[k] * np.eye(d), N_X)
                    # Compute divergences
                    W2_empirical[nr, i_d, k] = utils.wass_distance(X, Y, type='exact', order=1)
                    SW2[nr, i_d, k] = utils.sw_distance(X, Y, n_montecarlo=1, L=n_proj, p=1)
                    Sinkhorn[nr, i_d, k], _ = spc.sinkhorn_normalized(torch.FloatTensor(X), torch.FloatTensor(Y),
                                                                  eps, N_X, niter, p=1)
                    SlicedSinkhorn[nr, i_d, k], _ = utils.sliced_sinkhorn_torch(torch.FloatTensor(X), torch.FloatTensor(Y),
                                                                            eps, niter, n_proj=n_proj, p=1, computation="sequential")
                    MMD2[nr, i_d, k] = utils._mmd2(X, Y, type_kernel="rbf")
                    SlicedMMD2[nr, i_d, k] = utils.sliced_mmd2(X, Y, type_kernel="rbf", n_proj=n_proj)
        # Save distances
        with open(os.path.join(dirname, "W2_empirical"), "wb") as f:
            pickle.dump(W2_empirical, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(dirname, "SW2"), "wb") as f:
            pickle.dump(SW2, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(dirname, "Sinkhorn"), "wb") as f:
            pickle.dump(Sinkhorn, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(dirname, "SlicedSinkhorn"), "wb") as f:
            pickle.dump(SlicedSinkhorn, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(dirname, "MMD2"), "wb") as f:
            pickle.dump(MMD2, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(dirname, "SlicedMMD2"), "wb") as f:
            pickle.dump(SlicedMMD2, f, pickle.HIGHEST_PROTOCOL)

    # Plot results
    plot_distances(dims, sigmas2)
