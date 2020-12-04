## Statistical and Topological Properties of Sliced Probability Divergences

This repository contains the code to reproduce the experiments of [Statistical and Topological Properties of Sliced Probability Divergences](https://arxiv.org/abs/2003.05783), accepted as a spotlight presentation at NeurIPS 2020. 
Please cite our paper if you use any of our code. 

#### Requirements
Joblib, Matplotlib, Multiprocessing, Numpy, POT, PyTorch

#### Description of the .py files 

- `compare_dist.py` : Illustration of the topological result in Theorem 2. Run it to reproduce Figure 1.
- `study_complexity.py` : Illustration of the sample and projection complexity results for Sliced-Wasserstein and Sliced-Sinkhorn, on the synthetical setting. Run it to reproduce Figures 2 and 3 (main doc) and Figures S1 and S2 (supplementary doc). 
- `real_data_exp.py` : Two-sample testing problem (for data integration) on MNIST and CIFAR-10. Run it to reproduce Figure 4.
- `sinkhorn_pointcloud.py` : Contains the function that computes the optimal transport regularized cost and Sinkhorn divergences.
- `utils.py` : Contains the functions that compute different divergences and their sliced versions (Wasserstein, Sinkhorn, MMD).
