import numpy as np
import ot
import torch
import sinkhorn_pointcloud as spc
from joblib import Parallel, delayed
import multiprocessing


def wass_distance(X, Y, order=2, type='exact'):
    """
    Computes the (exact or approximate) Wasserstein distance of order 1 or 2 between empirical distributions
    """
    if order == 2:
        M = ot.dist(X, Y)
    elif order == 1:
        M = ot.dist(X, Y, metric='euclidean')
    else:
        raise Exception("Order should be 1 or 2.")
    a = np.ones((X.shape[0],))/X.shape[0]
    b = np.ones((Y.shape[0],))/Y.shape[0]
    if type == 'approx':
        # Regularized problem solved with Sinkhorn
        reg = 1
        return ot.sinkhorn2(a, b, M, reg)**(1/order)
    else:
        # Without regularization
        return ot.emd2(a, b, M)**(1/order)


def wass_gaussians(mu1, mu2, Sigma1, Sigma2):
    """
    Computes the Wasserstein distance of order 2 between two Gaussian distributions
    """
    d = mu1.shape[0]
    if d == 1:
        w2 = (mu1 - mu2)**2 + (np.sqrt(Sigma1) - np.sqrt(Sigma2))**2
    else:
        prodSigmas = Sigma2**(1/2)*Sigma1*Sigma2**(1/2)
        w2 = np.linalg.norm(mu1 - mu2)**2 + np.trace(Sigma1 + Sigma2 - 2*(prodSigmas)**(1/2))
    return np.sqrt(w2)


def sw_distance(X, Y, n_montecarlo=1, L=100, p=2):
    """
    Computes the Sliced-Wasserstein distance between empirical distributions
    """
    X = np.stack([X] * n_montecarlo)
    M, N, d = X.shape
    order = p
    # Project data
    theta = np.random.randn(M, L, d)
    theta = theta / (np.sqrt((theta ** 2).sum(axis=2)))[:, :, None]  # normalization (theta is in the unit sphere)
    theta = np.transpose(theta, (0, 2, 1))
    xproj = np.matmul(X, theta)
    yproj = np.matmul(Y, theta)
    # Compute percentiles
    T = 100
    t = np.linspace(0, 100, T + 2)
    t = t[1:-1]
    xqf = (np.percentile(xproj, q=t, axis=1))
    yqf = (np.percentile(yproj, q=t, axis=1))
    # Compute expected SW distance
    diff = (xqf - yqf).transpose((1, 0, 2))
    sw_dist = (np.abs(diff) ** order).mean()
    sw_dist = sw_dist ** (1/order)
    return sw_dist


def sw_gaussians(mu1, mu2, Sigma1, Sigma2, n_proj=100):
    """
    Computes the Sliced-Wasserstein distance of order 2 between two Gaussian distributions
    """
    d = mu1.shape[0]
    # Project data
    thetas = np.random.randn(n_proj, d)
    thetas = thetas / (np.sqrt((thetas ** 2).sum(axis=1)))[:, None]  # Normalize
    proj_mu1 = thetas @ mu1
    proj_mu2 = thetas @ mu2
    sw2 = 0
    for l in range(n_proj):
        th = thetas[l]
        proj_sigma1 = (th @ Sigma1) @ th
        proj_sigma2 = (th @ Sigma2) @ th
        sw2 += wass_gaussians(np.array([proj_mu1[l]]), np.array([proj_mu2[l]]),
                              np.array([proj_sigma1]), np.array([proj_sigma2]))**2
    sw2 /= n_proj
    return np.sqrt(sw2)


def sliced_sinkhorn_torch(X, Y, epsilon, niter, n_proj=100, p=2, computation="parallel"):
    """
    Computes the normalized Sliced-Sinkhorn divergence between empirical distributions
    """
    N, dn = X.shape
    M, dm = Y.shape
    assert dn == dm and M == N
    # Project data
    theta = torch.randn((n_proj, dn))
    theta = torch.stack([th / torch.sqrt((th ** 2).sum()) for th in theta])
    if len(theta.shape) == 1:
        xproj = torch.matmul(X, theta)
        yproj = torch.matmul(Y, theta)
    else:
        xproj = torch.matmul(X, theta.t())
        yproj = torch.matmul(Y, theta.t())
    # Compute Sinkhorn divergence between the projected distributions (parallel or sequential computation)
    if computation == "parallel":
        num_cores = multiprocessing.cpu_count()
        res = Parallel(n_jobs=num_cores, prefer="threads")(delayed(spc.sinkhorn_normalized)(xproj[:, l][:, np.newaxis], yproj[:, l][:, np.newaxis], epsilon, N, niter, p=p) for l in range(n_proj))
        res = list(zip(*res))
        ss_distances = list(res[0])
        niter_all = list(res[1])
        ss_dist = torch.mean(torch.stack(ss_distances) ** p)
        niter_avg = np.mean(niter_all)
    elif computation == "sequential":
        ss_dist = 0
        niter_avg = 0
        for l in range(n_proj):
            res = spc.sinkhorn_normalized(xproj[:, l][:, np.newaxis], yproj[:, l][:, np.newaxis], epsilon, N, niter, p=p)
            ss_dist += res[0] ** p
            niter_avg += res[1]
        ss_dist /= n_proj
        niter_avg /= n_proj
    else:
        raise Exception(
            "Option 'computation' must be 'parallel' or 'sequential', not '{}'.".format(computation))
    return ss_dist, niter_avg


def _mmd2(X, Y, type_kernel="energy"):
    """
        Computes the maximum mean discrepancy with the biased estimator
        Adapted from https://github.com/dougalsutherland/opt-mmd
    """
    if type_kernel == "rbf":
        from sklearn.metrics.pairwise import euclidean_distances
        sub = lambda feats, n: feats[np.random.choice(
            feats.shape[0], min(feats.shape[0], n), replace=False)]
        Z = np.r_[X, Y]
        D2 = euclidean_distances(Z, squared=True)
        upper = D2[np.triu_indices_from(D2, k=1)]
        kernel_width = np.median(upper, overwrite_input=True)
        bandwidth = np.sqrt(kernel_width / 2)
        del Z, D2, upper
        gamma = 1 / kernel_width
        XX = np.dot(X, X.T)
        XY = np.dot(X, Y.T)
        YY = np.dot(Y, Y.T)

        X_sqnorms = np.diagonal(XX)
        Y_sqnorms = np.diagonal(YY)

        K_XY = np.exp(-gamma * (
                -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
        K_XX = np.exp(-gamma * (
                -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
        K_YY = np.exp(-gamma * (
                -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    elif type_kernel == "energy":
        K_XY = - ot.dist(X, Y) / 2.
        K_XX = - ot.dist(X, X) / 2.
        K_YY = - ot.dist(Y, Y) / 2.
    else:
        raise Exception("The kernel type is not implemented.")
    # Compute the biased estimator of MMD
    mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return mmd2


def sliced_mmd2(X, Y, type_kernel="energy", n_proj=100):
    """
        Computes the sliced maximum mean discrepancy with the biased estimator
    """
    N, dn = X.shape
    M, dm = Y.shape
    assert dn == dm and M == N
    # Project data
    theta = np.random.randn(n_proj, dn)
    theta = np.stack([th / np.sqrt((th ** 2).sum()) for th in theta])
    xproj = np.matmul(X, theta.T)
    yproj = np.matmul(Y, theta.T)
    # Compute Sliced-MMD
    smmd2 = [_mmd2(xproj[:, l][:, np.newaxis], yproj[:, l][:, np.newaxis], type_kernel) for l in range(n_proj)]
    smmd2 = np.array(smmd2).mean()
    return smmd2
