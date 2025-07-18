# Coreset-based Variational GP (CVGP) regression

This is the codebase for our work on [Accurate and Scalable Stochastic Gaussian Process Regression via Learnable Coreset-based Variational Inference](https://openreview.net/forum?id=MCfTk4K1Ig), presented at [UAI2025](https://www.auai.org/uai2025/accepted_papers)

It provides a novel stochastic variational inference method for Gaussian process (GP) regression:
  - it defines a GP posterior over a learnable set of coresets, i.e., over pseudo-input/output, weighted pairs.
  - it reduces the dimensionality of the variational parameter search space to linear complexity, while ensuring numerical stability at SOTS time and space complexities

## CVGP

- CVGP is defined in terms of
  - the GP prior,
  - and the (weighted) data likelihood.

- CVGP is trained via variational inference
  - We derive a variational lower-bound on the log-marginal likelihood by marginalizing over the latent GP coreset variables
  - CVGP's lower-bound is amenable to stochastic optimization.

# Requirements

Using anacoda;

1. Create a new conda environment:

`conda create -n "cvgp_env" python=3.9.0 anaconda`

2. Iniate the environment:

`conda activate cvgp_env`

3. Install requirements:

`pip install -r ./requirements.txt`

4. Install torch (see: https://pytorch.org/):

__windows__: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`


__linux__: `pip3 install torch torchvision torchaudio`


__mac__: `pip3 install torch torchvision torchaudio`

5. Install UCI datasets:

`python -m pip install git+https://github.com/treforevans/uci_datasets.git`

6. Run:

`python main.py --epochs 1000 --ip_size 50 --loss CVTGP --cv_folds 5 --dataset parkinsons --check_stop 150 --early_stop 20 --save_metric rmse --device cuda`

(CVTGP in code -> CVGP in paper)
