import numpy as np
import sinkhorn_pointcloud as spc
import torch
import utils
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'
plt.rc('text', usetex=True)


def plot_complexity_sinkhorn(dirname, list_eps, list_dims, list_samples, n_runs, n_proj):
    with open(os.path.join(dirname, "sink_eps={}_dims={}_nruns={}".format(
            [i for i in list_eps], [i for i in list_dims], n_runs)), "rb") as f:
       Sinkhorn = pickle.load(f)
    with open(os.path.join(dirname, "slicedsink_eps={}_dims={}_nproj={}_nruns={}".format(
            [i for i in list_eps], [i for i in list_dims], n_proj, n_runs)), "rb") as f:
       SlicedSinkhorn = pickle.load(f)
    cmap = matplotlib.cm.get_cmap('Set1')  # color map

    # Study the influence of the data dimension on the sample complexity
    for e in range(list_eps.shape[0]):
        fig1 = plt.figure()
        eps = list_eps[e]
        for d in range(list_dims.shape[0]):
            dim = list_dims[d]
            sinkhorn_mean = Sinkhorn[:, e, d, :, 0].mean(axis=0)
            sinkhorn_10 = np.percentile(Sinkhorn[:, e, d, :, 0], 10, axis=0)
            sinkhorn_90 = np.percentile(Sinkhorn[:, e, d, :, 0], 90, axis=0)
            plt.loglog(list_samples, sinkhorn_mean, label=r"Sinkhorn, $d={}$".format(dim),
                      color=cmap(d), lw=1.5)
            plt.fill_between(list_samples, sinkhorn_10, sinkhorn_90, facecolor=cmap(d), alpha=0.2)
            slicedsinkhorn_mean = SlicedSinkhorn[:, e, d, :, 0].mean(axis=0)
            slicedsinkhorn_10 = np.percentile(SlicedSinkhorn[:, e, d, :, 0], 10, axis=0)
            slicedsinkhorn_90 = np.percentile(SlicedSinkhorn[:, e, d, :, 0], 90, axis=0)
            plt.loglog(list_samples, slicedsinkhorn_mean, label=r"Sliced-Sinkhorn, $d={}$".format(dim),
                       ls="--", lw=1.5, color=cmap(d))
            plt.fill_between(list_samples, slicedsinkhorn_10, slicedsinkhorn_90, facecolor=cmap(d), alpha=0.2)
            plt.legend()
            plt.title(r"$\varepsilon = {}$".format(eps))
            plt.xlabel("number of samples")
            plt.ylabel("divergence")
        fig1.savefig(os.path.join(dirname, "sink_complexity_eps={}_nproj={}_nruns={}.pdf".format(eps, n_proj, n_runs)), bbox_inches='tight')

    # Study the influence of the regularization on the sample complexity
    for d in range(list_dims.shape[0]):
        fig2 = plt.figure()
        dim = list_dims[d]
        for e in range(list_eps.shape[0]):
            eps = list_eps[e]
            sinkhorn_mean = Sinkhorn[:, e, d, :, 0].mean(axis=0)
            sinkhorn_10 = np.percentile(Sinkhorn[:, e, d, :, 0], 10, axis=0)
            sinkhorn_90 = np.percentile(Sinkhorn[:, e, d, :, 0], 90, axis=0)
            plt.loglog(list_samples, sinkhorn_mean, label=r"Sinkhorn, $\varepsilon={}$".format(eps),
                       color=cmap(e), lw=1.5)
            plt.fill_between(list_samples, sinkhorn_10, sinkhorn_90, facecolor=cmap(e), alpha=0.2)
            slicedsinkhorn_mean = SlicedSinkhorn[:, e, d, :, 0].mean(axis=0)
            slicedsinkhorn_10 = np.percentile(SlicedSinkhorn[:, e, d, :, 0], 10, axis=0)
            slicedsinkhorn_90 = np.percentile(SlicedSinkhorn[:, e, d, :, 0], 90, axis=0)
            plt.loglog(list_samples, slicedsinkhorn_mean, label=r"Sliced-Sinkhorn, $\varepsilon={}$".format(eps),
                       ls="--", lw=1.5, color=cmap(e))
            plt.fill_between(list_samples, slicedsinkhorn_10, slicedsinkhorn_90, facecolor=cmap(e), alpha=0.2)
            plt.legend()
            plt.title(r"$d = {}$".format(dim))
            plt.xlabel("number of samples")
            plt.ylabel("divergence")
        fig2.savefig(os.path.join(dirname, "sink_complexity_dim={}_nproj={}_nruns={}.pdf".format(dim, n_proj, n_runs)), bbox_inches='tight')

    # Study the convergence speed by plotting number of iterations reached at convergence vs. dimension
    list_samples = np.array([10, 100, 1000])
    for e in range(list_eps.shape[0]):
        fig3 = plt.figure()
        eps = list_eps[e]
        for ns in range(len(list_samples)):
            nsample = list_samples[ns]
            sinkhorn_niter_mean = Sinkhorn[:, e, :, ns, 1].mean(axis=0)
            sinkhorn_niter_10 = np.percentile(Sinkhorn[:, e, :, ns, 1], 10, axis=0)
            sinkhorn_niter_90 = np.percentile(Sinkhorn[:, e, :, ns, 1], 90, axis=0)
            plt.loglog(list_dims, sinkhorn_niter_mean, label=r"Sinkhorn, $N={}$".format(nsample),
                       color=cmap(ns), lw=1.5)
            plt.fill_between(list_dims, sinkhorn_niter_10, sinkhorn_niter_90, facecolor=cmap(ns), alpha=0.2)
            slicedsinkhorn_niter_mean = SlicedSinkhorn[:, e, :, ns, 1].mean(axis=0)
            slicedsinkhorn_niter_10 = np.percentile(SlicedSinkhorn[:, e, :, ns, 1], 10, axis=0)
            slicedsinkhorn_niter_90 = np.percentile(SlicedSinkhorn[:, e, :, ns, 1], 90, axis=0)
            plt.loglog(list_dims, slicedsinkhorn_niter_mean, label=r"Sliced-Sinkhorn, $N={}$".format(nsample), ls="--", lw=1.5, color=cmap(ns))
            plt.fill_between(list_dims, slicedsinkhorn_niter_10, slicedsinkhorn_niter_90, facecolor=cmap(ns), alpha=0.2)
        plt.legend()
        plt.title(r"$\varepsilon = {}$".format(eps))
        plt.xlabel("dimension")
        plt.xticks(list_dims, list_dims)
        plt.ylabel("number of iterations until convergence")
        fig3.savefig(os.path.join(dirname, "sink_niter_eps={}_nproj={}_nruns={}.pdf".format(eps, n_proj, n_runs)),
                     bbox_inches='tight')


def sample_complexity_sinkhorn(list_eps, list_dims, list_samples, n_runs, n_iter, n_proj):
    # Compute Sinkhorn and Sliced-Sinkhorn for different parameters
    Sinkhorn = np.zeros((n_runs, list_eps.shape[0], list_dims.shape[0], list_samples.shape[0], 2))
    SlicedSinkhorn = np.zeros((n_runs, list_eps.shape[0], list_dims.shape[0], list_samples.shape[0], 2))
    for nr in range(n_runs):
        print("Run {}...".format(nr+1))
        for e in range(list_eps.shape[0]):
            eps = list_eps[e]
            print("\t Epsilon: {}".format(eps))
            for d in range(list_dims.shape[0]):
                dim = list_dims[d]
                print("\t\t Dimension: {}".format(dim))
                for n in range(list_samples.shape[0]):
                    print("\t\t\t Number of samples: {}".format(list_samples[n]))
                    n_samples = list_samples[n]
                    X = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), n_samples)
                    Y = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), n_samples)
                    Sinkhorn[nr, e, d, n] = spc.sinkhorn_normalized(
                        torch.FloatTensor(X), torch.FloatTensor(Y), eps, n_samples, n_iter)
                    SlicedSinkhorn[nr, e, d, n] = utils.sliced_sinkhorn_torch(
                        torch.FloatTensor(X), torch.FloatTensor(Y), eps, n_iter, n_proj)
                    # Store results
                    with open(os.path.join(dirname, "sink_eps={}_dims={}_nruns={}".format(
                            [i for i in list_eps], [i for i in list_dims], n_runs)), "wb") as f:
                       pickle.dump(Sinkhorn, f, pickle.HIGHEST_PROTOCOL)
                    with open(os.path.join(dirname, "slicedsink_eps={}_dims={}_nproj={}_nruns={}".format(
                            [i for i in list_eps], [i for i in list_dims], n_proj, n_runs)), "wb") as f:
                       pickle.dump(SlicedSinkhorn, f, pickle.HIGHEST_PROTOCOL)


def plot_complexity_sw(dirname, list_proj, list_dims, list_samples, n_runs):
    with open(os.path.join(dirname, "w2_dims={}_nruns={}".format(
            [i for i in list_dims], n_runs)), "rb") as f:
       W2 = pickle.load(f)
    with open(os.path.join(dirname, "sw2_proj={}_dims={}_nruns={}".format(
            [i for i in list_proj], [i for i in list_dims], n_runs)), "rb") as f:
       SW2 = pickle.load(f)

    cmap = matplotlib.cm.get_cmap('Set1')  # color map

    # Study the influence of the data dimension on the sample complexity
    for nl in range(list_proj.shape[0]):
        fig1 = plt.figure()
        n_proj = list_proj[nl]
        for d in range(list_dims.shape[0] - 1):
            dim = list_dims[d]
            w2_mean = W2[:, d, :].mean(axis=0)
            w2_10 = np.percentile(W2[:, d, :], 10, axis=0)
            w2_90 = np.percentile(W2[:, d, :], 90, axis=0)
            plt.loglog(list_samples, w2_mean, label=r"W2, $d = {}$".format(dim), color=cmap(d), lw=1.5)
            plt.fill_between(list_samples, w2_10, w2_90, facecolor=cmap(d), alpha=0.2)
            sw2_mean = SW2[:, nl, d, :].mean(axis=0)
            sw2_10 = np.percentile(SW2[:, nl, d, :], 10, axis=0)
            sw2_90 = np.percentile(SW2[:, nl, d, :], 90, axis=0)
            plt.loglog(list_samples, sw2_mean, label=r"SW2, $d = {}$".format(dim), ls="--", lw=1.5, color=cmap(d))
            plt.fill_between(list_samples, sw2_10, sw2_90, facecolor=cmap(d), alpha=0.2)
            ax = plt.gca()
            handles, labels = ax.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0][0:3] == 'SW2'))
            plt.legend(handles, labels, ncol=2)
            plt.xlabel("number of samples")
            plt.ylabel("distance")
            fig1.savefig(os.path.join(dirname, "sw2_complexity_nproj={}_nruns={}.pdf".format(n_proj, n_runs)),
                         bbox_inches='tight')

    # Study the influence of the number of projections on the sample complexity
    for d in range(list_dims.shape[0]):
        fig2 = plt.figure()
        dim = list_dims[d]
        w2_mean = W2[:, d, :].mean(axis=0)
        w2_10 = np.percentile(W2[:, d, :], 10, axis=0)
        w2_90 = np.percentile(W2[:, d, :], 90, axis=0)
        plt.loglog(list_samples, w2_mean, label="W2", color=cmap(0), lw=1.5)
        plt.fill_between(list_samples, w2_10, w2_90, facecolor=cmap(0), alpha=0.2)
        for nl in range(list_proj.shape[0]):
            n_proj = list_proj[nl]
            sw2_mean = SW2[:, nl, d, :].mean(axis=0)
            sw2_10 = np.percentile(SW2[:, nl, d, :], 10, axis=0)
            sw2_90 = np.percentile(SW2[:, nl, d, :], 90, axis=0)
            plt.loglog(list_samples, sw2_mean, label=r"SW2, {} proj".format(n_proj), ls="--", lw=1.5,
                       color=cmap(nl+1))
            plt.fill_between(list_samples, sw2_10, sw2_90, facecolor=cmap(nl+1), alpha=0.2)
        plt.legend()
        plt.xlabel("number of samples")
        plt.ylabel("distance")
        plt.title(r"$d = {}$".format(dim))
        fig2.savefig(os.path.join(dirname, "sw2_complexity_dim={}_nruns={}.pdf".format(dim, n_runs)), bbox_inches='tight')

    # Study the influence of the dimension on the projection complexity
    for n in range(list_samples.shape[0]):
        n_samples = list_samples[n]
        fig3 = plt.figure()
        for d in range(list_dims.shape[0]-1):
            dim = list_dims[d]
            error = np.abs(SW2[:, :-1, d, n] - SW2[:, -1, d, n][:, np.newaxis])
            error_mean = error.mean(axis=0)
            error_10 = np.percentile(error, 10, axis=0)
            error_90 = np.percentile(error, 90, axis=0)
            plt.loglog(list_proj[:-1], error_mean, label=r"SW2, $d = {}$".format(dim), ls="--", lw=1.5,
                       color=cmap(d))
            plt.fill_between(list_proj[:-1], error_10, error_90, facecolor=cmap(d), alpha=0.2)
        plt.legend()
        plt.xlabel("number of projections")
        plt.ylabel("Monte Carlo error")
        plt.title(r"$n = {}$".format(n_samples))
        fig3.savefig(os.path.join(dirname, "sw2_complexity_n={}_nruns={}.pdf".format(n_samples, n_runs)),
                     bbox_inches='tight')


def sample_complexity_sw(list_proj, list_dims, list_samples, n_runs):
    # Compute Wasserstein and Sliced-Wasserstein for different parameters
    W2 = np.zeros((n_runs, list_dims.shape[0], list_samples.shape[0]))
    SW2 = np.zeros((n_runs, list_proj.shape[0], list_dims.shape[0], list_samples.shape[0]))
    for nr in range(n_runs):
        print("Run {}...".format(nr+1))
        for d in range(list_dims.shape[0]):
            dim = list_dims[d]
            print("\t\t Dimension: {}".format(dim))
            for n in range(list_samples.shape[0]):
                n_samples = list_samples[n]
                print("\t\t\t Number of samples: {}".format(n_samples))
                X = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), n_samples)
                Y = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), n_samples)
                W2[nr, d, n] = utils.wass_distance(X, Y, type='exact')
                # Store Wasserstein results
                with open(os.path.join(dirname, "w2_dims={}_nruns={}".format(
                        [i for i in list_dims], n_runs)), "wb") as f:
                   pickle.dump(W2, f, pickle.HIGHEST_PROTOCOL)
                for nl in range(list_proj.shape[0]):
                    n_proj = list_proj[nl]
                    print("\t\t\t\t Number of projections: {}".format(n_proj))
                    SW2[nr, nl, d, n] = utils.sw_distance(X, Y, n_montecarlo=1, L=n_proj)
                    # Store SW results
                    with open(os.path.join(dirname, "sw2_proj={}_dims={}_nruns={}".format(
                            [i for i in list_proj], [i for i in list_dims], n_runs)), "wb") as f:
                       pickle.dump(SW2, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    n_runs = 100  # number of runs

    # Create directory that will contain the results for Sliced-Wasserstein
    dirname = os.path.join("results", "complexity", "sw")
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Set parameters
    list_proj = np.array([1, 10, 100, 1000, 10000])  # number of projections in the Monte Carlo approximation
    list_dims = np.array([2, 10, 20, 50, 100, 1000])  # list of dimension values
    list_samples = np.array([10, 50, 100, 500, 1000])  # different number of samples for the generated datasets

    # Compute divergences and plot figures
    sample_complexity_sw(list_proj, list_dims, list_samples, n_runs)
    plot_complexity_sw(dirname, list_proj, list_dims, list_samples, n_runs)

    # Create directory that will contain the results for Sliced-Sinkhorn
    dirname = os.path.join("results", "complexity", "sink")
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Set parameters
    n_proj = 10  # number of projections for Sliced-Sinkhorn
    list_eps = np.array([0.05, 1, 10, 100])  # list of regularization coefficients
    list_dims = np.array([2, 10, 20, 50, 100])  # list of dimension values
    list_samples = np.array([10, 50, 100, 500, 1000])  # different number of samples for the generated datasets
    n_iter = 2000  # max. number of iterations for Sinkhorn's algorithm

    # Compute divergences and plot figures
    sample_complexity_sinkhorn(list_eps, list_dims, list_samples, n_runs, n_iter, n_proj)
    plot_complexity_sinkhorn(dirname, list_eps, list_dims, list_samples, n_runs, n_proj)
