# GPEstimators.jl

This Julia package runs a seasonal-trend decomposition (estimation and forecasting) of a univariate time series using Gaussian processes (GPs).


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


## Background

The goal of this package is to decompose a given time series $\{(t_n, y_n)\}_{n=1}^N$ (where $t_n, y_n \in \mathbb{R}$) into three components:
- Low-frequency nonstationary **trend** component (denoted $\bar x_n$)
- Medium-frequency stationary **seasonal** component (denoted $\tilde x_n$)
- High-frequency stationary **noise** component (denoted $\epsilon_n$)

We accomplish this by estimating a continuous model $f(t)$ for each component such that $\bar x(t_n) \equiv \bar x_n$, $\tilde x(t_n) \equiv \tilde x_n$ and $\epsilon(t_n) \equiv \epsilon_n$ for each $n=1,\dots,N$.

Mathematically, this can be written as the data model
$$y(t) := \bar x(t) + \tilde x(t) + \epsilon(t),$$
but it is important to note that while $y(t_n) \approx y_n$ generally, they need not be equivalent, for example, when filtering out outliers to ensure stationarity of the noise model $\epsilon(t)$.

Finally, we also derive a model $r(t)$ for the **rate** of change of the trend model, defined by the ordinary differential equation (ODE)
$$r(t) := \frac{\text d\bar x(t)}{\text dt}.$$

NB: We have used a slight abuse of notation for simplicity of explanation; the model $f(t)$ that is actually estimated is a GP (a _distribution_ of paths $\rho[f](t)$) with posterior mean and variance functions. Here, we use the GP and its posterior mean function interchangeably.

### Logarithmic data

The above description is for linear (additive) time series. For data in the logarithmic (multiplicative) scale where $y_n \geq 0$, the decomposition of the data model is the product
$$y(t) := \bar x(t) \cdot \tilde x(t) \cdot \epsilon(t),$$
and the rate $r(t)$ is the exponential growth rate defined by the ODE
$$\frac{\text d\bar x(t)}{\text dt} = r(t)\bar x(t).$$

One big advantages of using GPs for time series analysis is that the _data need not be uniformly sampled_!!! :) So you can feel free to remove unwanted data points or use noninteger times (or even have redundant data, i.e., multiple data for a single "time" point, especially for non time series applications).


## Implementation

The simplest use-case for this package is the linearly-scaled, uniformly-sampled time series $\{(n,y_n)\}$. For this, we simply call the main function `decompose()`:
```julia
comp = decompose(y)
```
By default the times $t_n = n$ are used. In order to specify (nonuniform) times, we use the optional second argument
```julia
comp = decompose(y, t)
```
We can run the analysis on a logarithmically-scaled series with the keyword argument
```julia
comp = decompose(y, t, logscale=true)
```

### Outputs

The results are contained in a `NamedTuple` object (called `comp` above). The primary results of interest, i.e., the estimated series $\{\bar x_n\}$, $\{\tilde x_n\}$, $\{\epsilon_n\}$ and $\{r_n\}$ (as well as the data model series $\{y(t_n)\}$), can be found in `comp.series`.

For example, the mean values for the GP corresponding to the trend, i.e., $\{\bar x_n\}$ can be accessed with
```julia
x̄ = comp.series.x̄.μ
```

Since the fitted models are actually GPs, the lower and upper credible interval (CI) values are also outputted:
```julia
l_x̄, u_x̄ = comp.series.x̄.l, comp.series.x̄.u
```
By default, we compute the 95% CI, but this can be changed, for example, to a 99% CI with the keyword argument
```julia
comp = decompose(y, t, β_ci=0.99)
```

### Recomputing series

Instead of rerunning the full decomposition, if we wish to only recompute the series using, for instance, at different times $\{t_m'\}_{m=1}^M$ with a 90% CI, we can use the `get_series()` helper function and the old decomposition results `comp`:
```julia
series′ = get_series(comp.ρ.marg, t′, β_ci=0.9)
```
where `comp.ρ.marg` is the collection of the marginal GP models $y(t)$, $\bar x(t)$, $\tilde x(t)$, $\epsilon(t)$ and $r(t)$. (For logarithmically-scaled data, _you must remember to pass the keyword argument_ `logscale=true`.)

NB: Any of the $t_m'$ can be outside of the range $[t_1, t_N]$ ("forecasting" in the case where $t_m' > t_N$), but one must be extremely careful in interpreting of the results of such extrapolations.


## References

### Code for optimization and differentiation

1. https://juliagaussianprocesses.github.io/Stheno.jl/dev/examples/extended_mauna_loa/
2. https://juliagaussianprocesses.github.io/Stheno.jl/dev/examples/differentiation/


### On GPs more generally

3. C. E. Rasmussen & C. K. I. Williams, Gaussian Processes for Machine Learning, the MIT Press, 2006, ISBN 026218253X. © 2006 Massachusetts Institute of Technology. [www.GaussianProcess.org/gpml](https://gaussianprocess.org/gpml/)
