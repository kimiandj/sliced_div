import numpy as np
import torch
import torchvision.datasets as datasets
import os
import pickle
import sinkhorn_pointcloud as spc
import utils
import timeit
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'
plt.rc('text', usetex=True)

timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""

if __name__ == "__main__":
    # Set parameters
    data = "cifar"  # mnist or cifar
    compute = True
    n_runs = 10
    eps = 1  # regularization coefficient
    n_proj = 10  # number of projection
    n_iter = 10000  # maximum number of iterations for Sinkhorn's algorithm
    if data == "mnist":
        list_samples = np.array([10, 100, 500, 1000, 2500])
    else:
        list_samples = np.array([10, 100, 500, 1000])

    # Create directory that will contain the results
    dirname = os.path.join("results", data + "_exp")
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    if compute:
        # Load dataset
        if data == "mnist":
            img_size = 28
            obs = datasets.MNIST(root='./data', train=True, download=True, transform=None)
            obs = np.array([np.array(obs[i][0]) for i in range(len(obs))])
            obs = obs.reshape((len(obs), img_size * img_size))
            obs = 2. * obs / 255. - 1.  # normalization
            obs2 = obs
        elif data == "cifar":
            img_size = 32
            obs = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
            obs = np.array([np.array(obs[i][0]) for i in range(len(obs))])
            obs = obs.reshape((len(obs), img_size * img_size * 3))
            obs = 2. * obs / 255. - 1.  # normalization
            obs2 = obs
        else:
            raise Exception("dataset not found: must be mnist or cifar!")

        # Compute divergences
        sinkhorn = np.zeros((n_runs, list_samples.shape[0], 2))
        sliced_sinkhorn = np.zeros((n_runs, list_samples.shape[0], 2))
        w2 = np.zeros((n_runs, list_samples.shape[0], 2))
        sw2 = np.zeros((n_runs, list_samples.shape[0], 2))
        for nr in range(n_runs):
            for n in range(list_samples.shape[0]):
                n_samples = list_samples[n]
                print("Number of samples:{}".format(n_samples))
                obs_n = obs[np.random.choice(len(obs), n_samples, replace=False)]
                obs2_n = obs2[np.random.choice(len(obs), n_samples, replace=False)]

                def compute_sinkhorn():
                    return spc.sinkhorn_normalized(
                        torch.FloatTensor(obs_n), torch.FloatTensor(obs2_n), eps, n_samples, n_iter)

                elapsed_time_res = timeit.timeit("compute_sinkhorn()", globals=globals(), number=1)  # measure execution time
                sinkhorn[nr, n] = elapsed_time_res[1][0]**(1/2), elapsed_time_res[0]

                def compute_sliced_sinkhorn():
                    return utils.sliced_sinkhorn_torch(torch.FloatTensor(obs_n), torch.FloatTensor(obs2_n), eps, n_iter, n_proj, computation="parallel")

                elapsed_time_res = timeit.timeit("compute_sliced_sinkhorn()", globals=globals(), number=1)  # measure execution time
                sliced_sinkhorn[nr, n] = elapsed_time_res[1][0]**(1/2), elapsed_time_res[0]

                def compute_w2():
                    return utils.wass_distance(obs_n, obs2_n, type='exact')

                elapsed_time_res = timeit.timeit("compute_w2()", globals=globals(), number=1)  # measure execution time
                w2[nr, n] = elapsed_time_res[1], elapsed_time_res[0]

                def compute_sw2():
                    return utils.sw_distance(obs_n, obs2_n, n_montecarlo=1, L=n_proj, p=2)

                elapsed_time_res = timeit.timeit("compute_sw2()", globals=globals(), number=1)  # measure execution time
                sw2[nr, n] = elapsed_time_res[1], elapsed_time_res[0]

                # Store results
                with open(os.path.join(dirname, "sink_eps={}".format(eps)), "wb") as f:
                    pickle.dump(sinkhorn, f, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(dirname, "slicedsink_eps={}_nproj={}".format(eps, n_proj)), "wb") as f:
                    pickle.dump(sliced_sinkhorn, f, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(dirname, "w2"), "wb") as f:
                    pickle.dump(w2, f, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(dirname, "sw2_nproj={}".format(n_proj)), "wb") as f:
                    pickle.dump(sw2, f, pickle.HIGHEST_PROTOCOL)
    else:
        # Load results
        with open(os.path.join(dirname, "sink_eps={}".format(eps)), "rb") as f:
            sinkhorn = pickle.load(f)
        with open(os.path.join(dirname, "slicedsink_eps={}_nproj={}".format(eps, n_proj)), "rb") as f:
            sliced_sinkhorn = pickle.load(f)
        with open(os.path.join(dirname, "w2"), "rb") as f:
            w2 = pickle.load(f)
        with open(os.path.join(dirname, "sw2_nproj={}".format(n_proj)), "rb") as f:
            sw2 = pickle.load(f)

    # Plot divergences vs. number of samples
    cmap = matplotlib.cm.get_cmap('Set1')  # color map
    fig1 = plt.figure(figsize=(7, 7))
    # Sinkhorn
    plt.loglog(list_samples, [sinkhorn[:, n, 0].mean(axis=0) for n in range(len(list_samples))], label="Sinkhorn", lw=3, color=cmap(0), marker=".", ms=10)
    plt.fill_between(list_samples, np.percentile(sinkhorn[:, :, 0], 10, axis=0), np.percentile(sinkhorn[:, :, 0], 90, axis=0), facecolor=cmap(0), alpha=0.2)
    # Sliced-Sinkhorn
    plt.loglog(list_samples, [sliced_sinkhorn[:, n, 0].mean(axis=0) for n in range(len(list_samples))], label="Sliced-Sink.",
               lw=3, color=cmap(0), ls="--", marker=".", ms=10)
    plt.fill_between(list_samples, np.percentile(sliced_sinkhorn[:, :, 0], 10, axis=0), np.percentile(sliced_sinkhorn[:, :, 0], 90, axis=0), facecolor=cmap(0), alpha=0.2)
    # Wasserstein
    plt.loglog(list_samples, [w2[:, n, 0].mean(axis=0) for n in range(len(list_samples))], label="Wasserstein", lw=3, color=cmap(1), marker=".", ms=10)
    plt.fill_between(list_samples, np.percentile(w2[:, :, 0], 10, axis=0), np.percentile(w2[:, :, 0], 90, axis=0), facecolor=cmap(1), alpha=0.2)
    # Sliced-Wasserstein
    plt.loglog(list_samples, [sw2[:, n, 0].mean(axis=0) for n in range(len(list_samples))], label="Sliced-Wass.", lw=3, color=cmap(1), ls="--", marker=".", ms=10)
    plt.fill_between(list_samples, np.percentile(sw2[:, :, 0], 10, axis=0), np.percentile(sw2[:, :, 0], 90, axis=0), facecolor=cmap(1), alpha=0.2)
    plt.legend(fontsize=18)
    plt.xlabel("number of samples", fontsize=28)
    plt.ylabel("divergence", fontsize=28)
    fig1.savefig(os.path.join(dirname, data + "_exp_eps={}_nproj={}_nruns={}.pdf".format(eps, n_proj, n_runs)),
                 bbox_inches='tight')

    # Plot execution time for Sinkhorn and Sliced-Sinkhorn vs. number of samples
    fig2 = plt.figure(figsize=(7, 7))
    plt.loglog(list_samples, [sinkhorn[:, n, 1].mean(axis=0) for n in range(len(list_samples))], label="Sinkhorn", lw=3, color=cmap(0), marker=".", ms=10)
    plt.fill_between(list_samples, np.percentile(sinkhorn[:, :, 1], 10, axis=0), np.percentile(sinkhorn[:, :, 1], 90, axis=0), facecolor=cmap(0), alpha=0.2)
    plt.loglog(list_samples, [sliced_sinkhorn[:, n, 1].mean(axis=0) for n in range(len(list_samples))], label="Sliced-Sinkhorn",
              lw=3, color=cmap(0), ls="--", marker=".", ms=10)
    plt.fill_between(list_samples, np.percentile(sliced_sinkhorn[:, :, 1], 10, axis=0), np.percentile(sliced_sinkhorn[:, :, 1], 90, axis=0), facecolor=cmap(0), alpha=0.2)
    plt.legend(fontsize=20)
    plt.xlabel("number of samples", fontsize=28)
    plt.ylabel("execution time (s)", fontsize=28)
    fig2.savefig(os.path.join(dirname, data + "_exp_timeplot_eps={}_nproj={}_nruns={}.pdf".format(eps, n_proj, n_runs)),
                 bbox_inches='tight')

