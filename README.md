# pd-scca
Primal-Dual Sparse Canonical Correlation Analysis

Code used to run the Primal-Dual SCCA method proposed in chapter 8 of [my thesis](http://discovery.ucl.ac.uk/10022771/). It requires the [`glmnet` package](https://web.stanford.edu/~hastie/glmnet_matlab/) to run.

The code is pretty general, i.e. it is possible to use it to run SCCA in the primal formulation for both `X` and `Y`. However, it can also be used to run one (or both views) in the dual formulation, by passing the `kernel_ridge_reg` function as string argument in `opt.regX` or `opt.regY`.
