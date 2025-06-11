# Coreset-based Variational GP (CVGP) regression

This is the codebase for our work on [Accurate and Scalable Stochastic Gaussian Process Regression via Learnable Coreset-based Variational Inference]()

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

