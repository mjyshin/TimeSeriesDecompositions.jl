# TimeSeriesDecompositions

This Julia package runs a seasonal-trend decomposition (estimation and forecasting) of a univariate time series using Gaussian processes (GPs) or the classical filters X-11 and STL.


## 1&ensp;Installation

Once you have installed [Julia](https://julialang.org/) (e.g., using [`juliaup`](https://github.com/JuliaLang/juliaup)), you can install this package from the Julia REPL by:

1. Pressing `]` to get to the Pkg REPL (and, optionally, activating your project-specific Julia environment)
2. Using the `dev` command (instead of `add` for registered packages)

```
(myenv) pkg> dev /path/to/TimeSeriesDecompositions
```

You can alternatively run the following short script directly from the Julia REPL:

```julia
using Pkg
Pkg.develop("/path/to/TimeSeriesDecompositions")
```

In both methods, *remember to replace the* `/path/to/` *with the actual path to this repository on your machine*! (You can drop the `/path/to/` if you place the package within `~/.julia/dev/`.) Also, you only have to do the above steps once (per project environment), after which you can use `using TimeSeriesDecompositions` to load it, as with any other Julia package.


## 2&ensp;Background

### 2.1&ensp;Seasonal-trend decomposition

The goal of this package is to decompose a given time series $\{(t_n, y_n)\}_{n=1}^N$ (where $t_n, y_n \in \mathbb{R}$) into three *additive* components:

- Low-frequency nonstationary **trend** component (denoted $\bar x_n$)
- Medium-frequency (non)stationary **seasonal** component (denoted $\tilde x_n$)
- High-frequency stationary **noise** component (denoted $\epsilon_n$)

We accomplish this by estimating a continuous model $f(t)$ for each component such that $\bar x(t_n) = \bar x_n$, $\tilde x(t_n) = \tilde x_n$ and $\epsilon(t_n) = \epsilon_n$ for each $n=1,\dots,N$.

Mathematically, this can be written as the data model

$$y(t) = \bar x(t) + \tilde x(t) + \epsilon(t).$$

Finally, we also derive a model $r(t)$ for the **rate** of change of the trend model, defined by the ordinary differential equation (ODE)

$$\frac{\text d\bar x(t)}{\text dt} = r(t).$$

**NB:** We have used a slight abuse of notation for simplicity of explanation; the model $f(t)$ that is actually estimated is a GP, a *distribution* of paths $\rho[f](t)$, with posterior mean $\mu_f(t)$ and standard deviation $\sigma_f(t)$ functions.

One major advantage of using GPs for time series analysis is that the *data need not be uniformly sampled*. So you can feel free to remove obvious outliers or use noninteger times (or even have redundant data, i.e., multiple $y(t)$ values for a single "time" point $t$)! ☺

### 2.2&ensp;Logarithmic data

The above description is for linear time series. For data in the logarithmic scale where $y(t) > 0$, the decomposition of the data model is the *product*

$$y(t) = \bar x(t) \cdot \tilde x(t) \cdot \epsilon(t),$$

and the rate $r(t)$ is the exponential growth rate defined by the ODE

$$\frac{\text d\bar x(t)}{\text dt} = r(t)\bar x(t).$$

We can run multiplicative decomposition by simply log-transforming the raw data to $\log y(t)$ then exponentiating the resulting component to $\exp f(t)$ (with the exception of the rate). If you have zeros in your log-scale data, you may consider adding a small fraction or use `log1p.(y)` and `expm1.(x̄)`.


## 3&ensp;Implementation

A decomposition is a *model*, an immutable struct of settings, fitted to data with `fit(model, y, t; period)`, as with the algorithms of [SciML](https://sciml.ai/) or the models of [MLJ](https://github.com/JuliaAI/MLJ.jl). There are three models:

- `GaussianProcess()`: Gaussian-process components with uncertainty, for any sampling
- `X11()`: X-11-style moving-average filters, for uniform sampling and an integer period
- `STL()`: a loess-based filter (a placeholder for the STL of Cleveland et al., 1990)

The simplest use-case is the linear-scale, uniformly-sampled time series $y_{1:N} := \{(n,y_n)\}_{n=1}^N$ with no seasonal pattern:

```julia
decomp = fit(GaussianProcess(), y)
```

By default, times $t_n = n$ are used. In order to specify (e.g., nonuniform) times, we use the optional second argument, and for a seasonal signal (e.g., annual seasonality with monthly data) the `period` keyword argument; the period describes the data, so it is given to `fit`, not to the model:

```julia
decomp = fit(GaussianProcess(), y, t; period=12)
```

The settings of a model are keyword arguments of its constructor, and `fit(M, y, t; period, settings...)` is shorthand for `fit(M(; settings...), y, t; period)`. The seasonal pattern may evolve slowly over time, over a timescale called the `bandwidth` (at least one period). Like the other hyperparameters, it is estimated from the data, including whether the pattern is static (`bandwidth=Inf`), which is kept unless a likelihood-ratio test favors an evolving pattern. As with any hyperparameter, you can give it a starting value in `θ_init` or fix it in `θ_fixed`:

```julia
decomp = fit(GaussianProcess(; θ_fixed=(bandwidth=36,)), y, t; period=12)   # evolves over ~3 years
decomp = fit(GaussianProcess(; θ_fixed=(bandwidth=Inf,)), y, t; period=12)  # static
```

The filter models are built from one smoother per stage, named `<component>_<stage>` in fitting order, and `trend` or `seasonal` sets every stage of a component:

```julia
model = X11(; trend_interm=Henderson(; bandwidth=12, endpoints=MinimumRevision(; ratio=0)))
decomp = fit(model, y; period=7)
decomp.smoother.trend_interm    # the fitted smoother of that stage
```

A model can be reused across series, and changed with [Accessors.jl](https://github.com/JuliaObjects/Accessors.jl): `@set model.trend_final.bandwidth = 16`.

### 3.1&ensp;Outputs

The results are contained in a `Decomposition` struct, which keeps the `model` it was fitted with. The primary results of interest, i.e., the estimated series $\bar x_{1:N}$, $\tilde x_{1:N}$, $\epsilon_{1:N}$ and $r_{1:N}$ can be found in `decomp.trend`, `decomp.seasonal`, `decomp.noise` and `decomp.rate`, respectively, and the fitted hyperparameters in `decomp.θ`.

For example, the mean values for the GP corresponding to the trend, i.e., $\bar x_{1:N}$ can be accessed with

```julia
x̄ = decomp.trend.mean
```

The standard deviation as well as the lower and upper credible interval (CI) values are also outputted (the filter models give point estimates only):

```julia
σ_x̄ = decomp.trend.std
l_x̄, u_x̄ = decomp.trend.lower, decomp.trend.upper
```

By default, we output the 50% and 90% CIs.

### 3.2&ensp;Recomputing decompositions

Instead of refitting a full decomposition, perhaps we wish to simply get the decomposition values (using the same training data) at different CIs or at different times $\{t_m'\}_{m=1}^M$. In this case, we can use the `predict(decomp, ...)` function, where `decomp` is the previously-trained decomposition results:

```julia
decomp′ = predict(decomp; sig=[0.68, 0.95])
decomp′ = predict(decomp, t′)
```

To refit new data with the same hyperparameters, fix them at the fitted values: `fit(GaussianProcess(; θ_fixed=decomp.θ), y′, t′; period=12)`.

**NB:** For `GaussianProcess`, any of the $t_m'$ can be outside of the range $[t_1, t_N]$ ("forecasting" in the case where $t_m' > t_N$), but one must be careful in interpreting of the results of such extrapolations, as one does with methods like (S)ARIMA. The filter models predict within the observed range only.


## References

The GP model is built on [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl) and [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl), with hyperparameters fitted by [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) using the exact gradient of the marginal likelihood.

1. C. E. Rasmussen & C. K. I. Williams, Gaussian Processes for Machine Learning, the MIT Press, 2006, ISBN 026218253X. © 2006 Massachusetts Institute of Technology. [www.GaussianProcess.org/gpml](https://gaussianprocess.org/gpml/)
