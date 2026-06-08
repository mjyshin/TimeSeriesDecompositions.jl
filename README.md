# GPDecompositions

This Julia package runs a seasonal-trend decomposition (estimation and forecasting) of a univariate time series using Gaussian processes (GPs).


## Installation

Once you have installed [Julia](https://julialang.org/) (e.g., using [`juliaup`](https://github.com/JuliaLang/juliaup)), you can install this package from the Julia REPL by:

1. Pressing `]` to get to the Pkg REPL (and, optionally, activating your project-specific Julia environment)
2. Using the `dev` command (instead of `add` for registered packages)

```
(myenv) pkg> dev /path/to/GPDecompositions
```

You can alternatively run the following short script directly from the Julia REPL:

```julia
using Pkg
Pkg.develop("/path/to/GPDecompositions")
```

In both methods, *remember to replace the* `/path/to/` *with the actual path to this repository on your machine*! (You can drop the `/path/to/` if you place the package within `~/.julia/dev/`.) Also, you only have to do the above steps once (per project environment), after which you can use `using GPDecompositions` to load it, as with any other Julia package.


## Background

The goal of this package is to decompose a given time series $\{(t_n, y_n)\}_{n=1}^N$ (where $t_n, y_n \in \mathbb{R}$) into three components:

- Low-frequency nonstationary **trend** component (denoted $\bar x_n$)
- Medium-frequency stationary **seasonal** component (denoted $\tilde x_n$)
- High-frequency stationary **noise** component (denoted $\epsilon_n$)

We accomplish this by estimating a continuous model $f(t)$ for each component such that $\bar x(t_n) = \bar x_n$, $\tilde x(t_n) = \tilde x_n$ and $\epsilon(t_n) = \epsilon_n$ for each $n=1,\dots,N$.

Mathematically, this can be written as the data model

$$y(t) = \bar x(t) + \tilde x(t) + \epsilon(t).$$

Finally, we also derive a model $r(t)$ for the **rate** of change of the trend model, defined by the ordinary differential equation (ODE)

$$\frac{\text d\bar x(t)}{\text dt} = r(t).$$

NB: We have used a slight abuse of notation for simplicity of explanation; the model $f(t)$ that is actually estimated is a GP, a _distribution_ of paths $\rho[f](t)$, with posterior mean $\mu_f(t)$ and standard deviation $\sigma_f(t)$ functions.

### Logarithmic data

The above description is for linear (additive) time series. For data in the logarithmic (multiplicative) scale where $y_n \geq 0$, the decomposition of the data model is the product

$$y(t) = \bar x(t) \cdot \tilde x(t) \cdot \epsilon(t),$$

and the rate $r(t)$ is the exponential growth rate defined by the ODE

$$\frac{\text d\bar x(t)}{\text dt} = r(t)\bar x(t).$$

We can 

One big advantages of using GPs for time series analysis is that the _data need not be uniformly sampled_!!! ☺ So you can feel free to remove obvious outliers or use noninteger times (or even have redundant data, i.e., multiple data for a single "time" point).


## Implementation

The simplest use-case for this package is the linearly-scaled, uniformly-sampled time series $\{(n,y_n)\}$ with no seasonal patterns. For this, we simply call the main function `fit(Decomposition, ...)` following the `MultivariateStats` API:

```julia
decomp = fit(Decomposition, y)
```

By default, times $t_n = n$ are used. In order to specify (nonuniform) times, we use the optional second argument

```julia
decomp = fit(Decomposition, y, t)
```

If your data has a seasonal signal, e.g., annual seasonality with monthly data, we can specify the `period` keyword argument

```julia
decomp = fit(Decomposition, y, t; period=12)
```

If the seasonality slowly evolves with time, we can specify the `bandwidth` keyword argument

```julia
decomp = fit(Decomposition, y, t; period=12, bandwidth=36)
```

### Outputs

The results are contained in a `Decomposition` struct. The primary results of interest, i.e., the estimated series $\{\bar x_n\}$, $\{\tilde x_n\}$, $\{\epsilon_n\}$ and $\{r_n\}$ can be found in `decomp.trend`, `decomp.seasonal`, `decomp.noise`, `decomp.rate`, respectively.

For example, the mean values for the GP corresponding to the trend, i.e., $\{\bar x_n\}$ can be accessed with

```julia
x̄ = decomp.trend.mean
```

The standard deviation as well as the lower and upper credible interval (CI) values are also outputted:

```julia
σ_x̄ = decomp.trend.std
l_x̄, u_x̄ = decomp.trend.lower, decomp.trend.upper
```

By default, we compute the 50% and 90% CIs, but this could be changed during the fitting, for example, to a 99% CI with the `sig` keyword argument:

```julia
decomp = fit(Decomposition, y, t; sig=[0.99])
```

### Recomputing decompositions

Instead of refitting a full decomposition (to fit to new data), perhaps we wish to simply recompute the decomposition at different CIs or at different times $\{t_m'\}_{m=1}^M$. In this case, we can use the `predict(decomp, ...)` function, where `decomp` is the old decomposition results:

```julia
decomp′ = predict(decomp; sig=[0.68, 0.95])
decomp′ = predict(decomp, t′)
```

NB: Any of the $t_m'$ can be outside of the range $[t_1, t_N]$ ("forecasting" in the case where $t_m' > t_N$), but one must be careful in interpreting of the results of such extrapolations.


## References

### Code for optimization and differentiation

1. https://juliagaussianprocesses.github.io/Stheno.jl/dev/examples/extended_mauna_loa/
2. https://juliagaussianprocesses.github.io/Stheno.jl/dev/examples/differentiation/

### On GPs more generally

3. C. E. Rasmussen & C. K. I. Williams, Gaussian Processes for Machine Learning, the MIT Press, 2006, ISBN 026218253X. © 2006 Massachusetts Institute of Technology. [www.GaussianProcess.org/gpml](https://gaussianprocess.org/gpml/)
