# FiniteMC

This repository contains code and notes for a research project conducted under the supervision of **Peter Glynn** on **error analysis for estimating stationary distributions of Markov chains with compact, continuous state spaces**.

## Project focus

The goal of this project is to understand and quantify the approximation error that arises when computing or estimating a stationary distribution for a Markov chain evolving on a compact continuous domain (e.g. $(0,1)$ or $[0,1]^d$). We study both methodological and numerical sources of error, including discretization bias and Monte Carlo / quasi–Monte Carlo integration error.

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

## Repository contents

- Source code for the Markov chain models and estimators.
- Implementations of discretization-based and sampling-based stationary distribution approximations.
- Experiments, diagnostics, and error comparisons between methods.
- Supporting notes and derivations.

## Status

Work in progress — results and implementations will be updated as the project develops.
