const _default_optimizer = LBFGS(;
    alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
    linesearch=Optim.LineSearches.BackTracking(),
)

function _nll(θ::NamedTuple, t_obs, y_obs::AbstractVector{<:Real}, kernels::NamedTuple)
    gp = _build_gp(θ, kernels)
    return -logpdf(gp(t_obs, 1e-12), y_obs)
end

function _optimize_params(
    θ₀::NamedTuple, fix::AbstractVector{Symbol},
    t_obs, y_obs::AbstractVector{<:Real};
    kernels::NamedTuple=_default_kernels,
    optimizer=_default_optimizer, max_iter::Int=1_000,
)
    wrapped = _init_params(θ₀, fix)
    θ₀_flat, unflatten = ParameterHandling.value_flatten(wrapped)
    isempty(θ₀_flat) && return unflatten(θ₀_flat)

    loss = x -> _nll(unflatten(x), t_obs, y_obs, kernels)

    function g!(G, x)
        G .= only(Zygote.gradient(loss, x))
    end

    options = Optim.Options(; iterations=max_iter, show_trace=false)
    result = optimize(loss, g!, θ₀_flat, optimizer, options)
    return unflatten(result.minimizer)
end

function _fit_gp(
    y::AbstractVector{<:Real}, t::AbstractVector{<:Real};
    period::Real=Inf,
    bandwidth::Real=Inf,
    θ_init::Union{Nothing, NamedTuple}=nothing,
    θ_fixed::AbstractVector{Symbol}=Symbol[],
    kernels::NamedTuple=_default_kernels,
    optimizer=_default_optimizer,
    max_iter::Int=1_000,
)
    y_fit, ȳ, s_y = standardize(y)

    # Defaults are all-ones in standardized space. User θ_init is in original units;
    # _merge_params divides σ fields by s_y before merging.
    θ0_base = _merge_params(θ_init, s_y)
    period_eff = θ0_base.σ_seasonal == 0.0 ? Inf : period
    θ0 = (θ0_base..., period=Float64(period_eff), bandwidth=Float64(bandwidth))

    t_obs = GPPPInput(:y, collect(Float64, t))
    θ_opt = _optimize_params(θ0, θ_fixed, t_obs, Float64.(y_fit); kernels, optimizer, max_iter)

    gp = _build_gp(θ_opt, kernels)
    posterior_gp = posterior(gp(t_obs, 1e-12), Float64.(y_fit))

    θ_final = _destandardize_params(θ_opt, s_y)

    return (posterior=posterior_gp, θ=θ_final, ȳ=ȳ, s_y=s_y)
end

"""
    fit(Decomposition, y, t; kwargs...) -> Decomposition

Decompose time series `y` observed at times `t` into trend, seasonal, and noise
components using a Gaussian process model.

`y` is internally z-score standardized before fitting; all returned parameters
and component estimates are in the original scale.

# Keyword arguments
- `period`: seasonal period in the same units as `t`. `Inf` (default) disables
  the seasonal component. If `θ_init` contains `:period`, that value is used
  as the default (explicit `period` kwarg still overrides).
- `bandwidth`: temporal envelope lengthscale for quasi-periodic seasonality.
  `Inf` (default) gives a stationary periodic kernel. A finite value lets the
  seasonal pattern evolve slowly over time. Never optimized. If `θ_init`
  contains `:bandwidth`, that value is used as the default.
- `kernel`: `NamedTuple` specifying the kernel for each component. Supported keys:
  `trend`, `seasonal`, `bandwidth`. Unspecified keys default to `SEKernel()`.
  Example: `kernel=(trend=Matern52Kernel(), seasonal=Matern32Kernel())`.
- `approx_rate`: use fast finite-difference rate with `std=NaN` (default `false`);
  set `true` to skip the posterior covariance computation for speed
- `θ_init`: `NamedTuple` of initial values for the optimized hyperparameters
  `(σ_trend, l_trend, σ_seasonal, l_seasonal, σ_noise)`. Boundary values
  (`σ=0`, `l=Inf`) are automatically fixed.
- `θ_fixed`: `Vector{Symbol}` of optimized hyperparameters to hold fixed
- `sig`: credible interval levels, e.g. `[0.5, 0.9]` (default)
- `t_out`: output time grid; defaults to `t`
- `optimizer`: Optim optimizer (default L-BFGS with backtracking)
- `max_iter`: maximum optimizer iterations (default 1000)
"""
function fit(
    ::Type{Decomposition},
    y::AbstractVector{<:Real},
    t::AbstractVector{<:Real}=eachindex(y);
    period::Union{Real,Nothing}=nothing,
    bandwidth::Union{Real,Nothing}=nothing,
    kernel::Union{Nothing, NamedTuple}=nothing,
    approx_rate::Bool=false,
    θ_init::Union{Nothing, NamedTuple}=nothing,
    θ_fixed::AbstractVector{<:Union{Symbol,String}}=Symbol[],
    sig::AbstractVector{<:Real}=[0.5, 0.9],
    t_out::AbstractVector{<:Real}=t,
    optimizer=_default_optimizer,
    max_iter::Int=1_000,
)
    # Explicit kwargs take precedence; θ_init fields override the default (Inf).
    period_eff    = period    !== nothing ? Float64(period)    :
                    (θ_init !== nothing && haskey(θ_init, :period))    ? Float64(θ_init.period)    : Inf
    bandwidth_eff = bandwidth !== nothing ? Float64(bandwidth) :
                    (θ_init !== nothing && haskey(θ_init, :bandwidth)) ? Float64(θ_init.bandwidth) : Inf

    length(y) == length(t) || throw(DimensionMismatch("y and t must have the same length"))
    all(0 .< sig .< 1)     || throw(DomainError(sig, "sig levels must be in (0, 1)"))
    isfinite(period_eff)    && period_eff    <= 0 && throw(DomainError(period_eff,    "period must be positive"))
    isfinite(bandwidth_eff) && bandwidth_eff <= 0 && throw(DomainError(bandwidth_eff, "bandwidth must be positive"))

    perm    = sortperm(t)
    y_s     = Float64.(y[perm])
    t_s     = Float64.(t[perm])
    kernels = _merge_kernels(kernel)

    result = _fit_gp(y_s, t_s;
        period=period_eff,
        bandwidth=bandwidth_eff,
        θ_init,
        θ_fixed=Symbol.(θ_fixed),
        kernels,
        optimizer,
        max_iter,
    )

    return _build_decomposition(
        result.posterior, Float64.(t_out), result.θ,
        result.ȳ, result.s_y;
        sig=Float64.(sig), approx_rate,
    )
end
