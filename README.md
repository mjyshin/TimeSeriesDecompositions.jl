# GPEstimators.jl

This Julia package runs a seasonal-trend decomposition (estimation and forecasting) of time series using Gaussian processes (GPs).


## Installation

Once you have installed [Julia](https://julialang.org/) (I personally recommend using [`juliaup`](https://github.com/JuliaLang/juliaup)), you can install this package from the Julia REPL by:

1. Pressing `]` to get to the Pkg REPL (and, optionally, activating your project-specific Julia environment)
2. Using the `dev` command (instead of `add` for registered packages)
```
(@v1.9) pkg> dev /path/to/RealTimeEstimators.jl
```

You can alternatively run the following short script directly from the Julia REPL:
```julia
using Pkg
Pkg.develop("/path/to/GPEstimators.jl")
```

In both methods, *remember to replace the* `/path/to/` *with the actual path to this repository on your machine*! Also, you only have to do the above steps once (per project environment), after which you can use `using GPEstimators` to load it, as with any other Julia package.


## References

### Code for optimization and differentiation

https://juliagaussianprocesses.github.io/Stheno.jl/dev/examples/extended_mauna_loa/
https://juliagaussianprocesses.github.io/Stheno.jl/dev/examples/differentiation/


### On GPs more generally

C. E. Rasmussen & C. K. I. Williams, Gaussian Processes for Machine Learning, the MIT Press, 2006, ISBN 026218253X. © 2006 Massachusetts Institute of Technology. [www.GaussianProcess.org/gpml](https://gaussianprocess.org/gpml/)
