# FiniteMC

This repository contains code and notes for a research project conducted under the supervision of **Peter Glynn** on **error analysis for estimating stationary distributions of Markov chains with compact, continuous state spaces**.

## Project focus

The goal of this project is to understand and quantify the approximation error that arises when computing or estimating a stationary distribution for a Markov chain evolving on a **compact (closed and bounded)** continuous domain (e.g. $[0,1]$ or $[0,1]^d$). We study both methodological and numerical sources of error, including discretization bias and Monte Carlo / quasi–Monte Carlo integration error.

## Methods studied

Two main approaches are analyzed:

1. **Probabilistic grid discretization**
   - Discretize the continuous state space using a grid.
   - Approximate the Markov transition operator by computing (or estimating) transition probabilities between grid cells.
   - Use the resulting finite-state Markov chain to approximate the stationary distribution of the original chain.

2. **Sampling-based estimation using Sobol sequences (QMC)**
   - Estimate integrals and transition expectations using **Sobol low-discrepancy sequences**.
   - Compare standard Monte Carlo with **quasi–Monte Carlo (QMC)** schemes in terms of convergence and error behavior.
   - Use QMC to reduce variance / integration error in stationary distribution estimation.

## Current progress

- Implemented a 1D testbed Markov chain with an explicit transition kernel and known stationary distribution (Beta), enabling controlled error studies.
- Built a grid-based discretization pipeline:
  - Gauss–Legendre quadrature over grid cells to approximate transition probabilities,
  - computation of the discrete stationary distribution as the eigenvector of $P^\top$ corresponding to eigenvalue $1$,
  - conversion of cell masses into a piecewise-constant density estimate.
- Added an evaluation framework based on test functions (rewards) such as moments, boundary-weighted polynomials, and $\log$-type functions to probe boundary sensitivity.
- Ran convergence experiments over grid resolution $h$ and produced log–log error plots against $1/h$.

![1D Beta-binomial discretization error](1DBetaBinErrors.png)

## Repository contents

- Source code for Markov chain models and estimators.
- Implementations of discretization-based and sampling-based stationary distribution approximations.
- Experiments, diagnostics, and error comparisons between methods.
- Supporting notes and derivations.

## Status

Work in progress — results and implementations will be updated as the project develops.
