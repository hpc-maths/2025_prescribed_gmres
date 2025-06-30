# Prescription of convergence curves for weighted and preconditioned GMRES

This repository contains the Matlab scripts associated with the paper 

> *Any nonincreasing convergence curves are simultaneously possible for GMRES and weighted GMRES, as well as for left and right preconditioned GMRES*, P. Matalon, N. Spillane, 2025.

The scripts hold the implementation of the theorems' constructive proofs, and reproduce the numerical experiments. 

## Installation

Simply clone the repository.

Note that the Matlab package [krylov4r](https://github.com/hpc-maths/krylov4r) is used as a dependency. However, it is provided here, so all scripts can be run without any further installation.

## Description

All runnable scripts are in `src/`:
```matlab
generate_system_prescribed_cc.m          % proof of Theorem 9
norm_for_prescribed_convergence.m        % proofs of Theorems 11 and 21
generate_system_weighted_prescribed_cc.m % proof of Corollary 14
generate_system_LR_prec_prescribed_cc.m  % proof of Theorem 22
inverse_left_right_prec_cc.m             % proof of Corollary 24
random_T_prescribed_g_and_mu.m           % illustrations (section 4.1)
ipsen.m                                  % illustrations (section 4.2)
difference_left_right_prec.m             % illustrations (section 4.3)
```

## Authors
Pierre Matalon, Nicole Spillane
