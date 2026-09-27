# Diagonal jitter (standardized space) for the training covariance only; predictive
# covariances keep 1e-12 as the rate finite differences divide by 4h²
const _JITTER = 1e-6

const _default_optimizer = LBFGS(;
    alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
    linesearch=Optim.LineSearches.BackTracking(),
)

function _nll(θ::NamedTuple, t_obs, y_obs::AbstractVector{<:Real}, kernels::NamedTuple)
    gp = _build_gp(θ, kernels)
    return -logpdf(gp(t_obs, _JITTER), y_obs)
end

function _optimize_params(
    θ₀::NamedTuple,
    fix::AbstractVector{Symbol},
    t_obs,
    y_obs::AbstractVector{<:Real};
    kernels::NamedTuple=_default_kernels,
    optimizer=_default_optimizer,
    max_iter::Int=1_000,
)
    wrapped = _init_params(θ₀, fix)
    θ₀_flat, unflatten = ParameterHandling.value_flatten(wrapped)
    free = [k for (k, v) in pairs(wrapped) if !(v isa ParameterHandling.Fixed)]
    isempty(θ₀_flat) && return (θ=unflatten(θ₀_flat), free=free, result=nothing)

    loss = x -> _nll(unflatten(x), t_obs, y_obs, kernels)

    function g!(G, x)
        return G .= only(Zygote.gradient(loss, x))
    end

    options = Optim.Options(; iterations=max_iter, show_trace=false)
    result = optimize(loss, g!, θ₀_flat, optimizer, options)
    return (θ=unflatten(result.minimizer), free=free, result=result)
end

function _fit_gp(
    y::AbstractVector{<:Real},
    t::AbstractVector{<:Real};
    period::Real=Inf,
    bandwidth::Real=Inf,
    θ_init::Union{Nothing,NamedTuple}=nothing,
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
    l_trend_min = _trend_lower_bound(period_eff, bandwidth)
    l_trend₀ = _initial_l_trend(
        θ0_base.l_trend,
        θ_init !== nothing && haskey(θ_init, :l_trend),
        :l_trend ∈ θ_fixed,
        l_trend_min,
    )
    θ0 = merge(
        θ0_base,
        (l_trend=l_trend₀, period=Float64(period_eff), bandwidth=Float64(bandwidth)),
    )

    t_obs = GPPPInput(:y, collect(Float64, t))
    opt = _optimize_params(
        θ0, θ_fixed, t_obs, Float64.(y_fit); kernels, optimizer, max_iter
    )
    θ_opt = opt.θ

    gp = _build_gp(θ_opt, kernels)
    fx = gp(t_obs, _JITTER)
    posterior_gp = posterior(fx, Float64.(y_fit))

    at_bound = :l_trend ∈ opt.free && θ_opt.l_trend < (1 + 1e-3) * l_trend_min
    at_bound && @warn "l_trend is at its lower bound max(bandwidth, period) = " *
        "$l_trend_min: the data favour a trend that varies faster than the bound " *
        "allows, and that variation is left to the seasonal and noise components" *
        (isfinite(bandwidth) ? "; consider a smaller bandwidth" : "")
    residual_structure = _residual_structure(posterior_gp, θ_opt, t, l_trend_min)
    residual_structure && @warn "the noise component is positively autocorrelated: " *
        "structure faster than max(bandwidth, period) = $l_trend_min is left in the noise"
    :l_seasonal ∈ opt.free &&
        θ_opt.l_seasonal > (1 - 1e-3) * _L_SEASONAL_MAX &&
        @info "l_seasonal is at its cap $_L_SEASONAL_MAX: the seasonal pattern is " *
            "essentially a pure sinusoid"

    θ_final = _destandardize_params(θ_opt, s_y)
    fit_info = (;
        kernels,
        free=opt.free,
        l_trend_min,
        at_bound,
        residual_structure,
        # log marginal likelihood of the original (unstandardized) data
        loglik=logpdf(fx, Float64.(y_fit)) - length(y) * log(s_y),
        optimizer=isnothing(opt.result) ? nothing : Optim.summary(opt.result),
        converged=isnothing(opt.result) || Optim.converged(opt.result),
        iterations=isnothing(opt.result) ? 0 : Optim.iterations(opt.result),
    )

    return (posterior=posterior_gp, θ=θ_final, ȳ=ȳ, s_y=s_y, fit_info=fit_info)
end

# Whether the lag-1 autocorrelation of the posterior-mean noise exceeds 2/√N. Fitted
# residuals are biased towards negative autocorrelation, so this errs on the quiet side.
# The lag is by index, which is a lag in time only for regularly sampled t.
function _residual_structure(post, θ::NamedTuple, t::AbstractVector, l_trend_min::Real)
    (l_trend_min > 0 && θ.σ_noise > 0 && length(t) > 2) || return false
    ϵ = mean(post(GPPPInput(:ϵ, collect(Float64, t)), 1e-12))
    ϵ = ϵ .- mean(ϵ)
    r₁ = sum(ϵ[1:(end - 1)] .* ϵ[2:end]) / sum(abs2, ϵ)
    return r₁ > 2 / sqrt(length(t))
end

"""
    fit(GPDecomp, y, t; kwargs...) -> GPDecomp

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
  seasonal pattern evolve slowly over time. Must be at least `period`. Never
  optimized. If `θ_init` contains `:bandwidth`, that value is used as the default.
- `kernel`: base kernel(s) of the components (default `Matern52Kernel()`). A single
  kernel applies to every component. A `NamedTuple` with keys `trend` and `seasonal`
  sets them separately, and `seasonal` may itself be a kernel (for both the periodic
  effect and its envelope) or a `NamedTuple` with keys `effect` and `envelope`.
  Unspecified components default to `Matern52Kernel()`. Examples:
  `kernel=SEKernel()`, `kernel=(trend=SEKernel(), seasonal=Matern32Kernel())`,
  `kernel=(seasonal=(effect=Matern32Kernel(), envelope=SEKernel()),)`.
- `approx_rate`: use fast finite-difference rate with `std=NaN` (default `false`);
  set `true` to skip the posterior covariance computation for speed
- `θ_init`: `NamedTuple` of initial values for the optimized hyperparameters
  `(σ_trend, l_trend, σ_seasonal, l_seasonal, σ_noise)`. Boundary values
  (`σ=0`, `l_trend=Inf`) are automatically fixed.
- `θ_fixed`: `Vector{Symbol}` of optimized hyperparameters to hold fixed
- `sig`: credible interval levels, e.g. `[0.5, 0.9]` (default)
- `t_out`: output time grid; defaults to `t`
- `optimizer`: Optim optimizer (default L-BFGS with backtracking)
- `max_iter`: maximum optimizer iterations (default 1000)

# Identifiability
The lengthscales are kept ordered from slowest to fastest,
`l_trend > max(bandwidth, period) ≥ bandwidth ≥ period > seasonal timescale > 0 (noise)`:
`l_trend` is optimized strictly above `max(bandwidth, period)` (above `period` if
`bandwidth = Inf`; unbounded without seasonality), and a fixed `l_trend` must satisfy
the bound. A warning is emitted if the optimum lies on the bound, or if the noise
component is positively autocorrelated at lag 1 (faster structure left in the noise;
the lag is by index, so this check assumes roughly regular sampling). The
seasonal kernel has its mean over one period removed, so the seasonal component averages
to zero over each period and cannot absorb the trend. `l_seasonal` is capped at 10,
beyond which the seasonal kernel is a pure sinusoid.
"""
function fit(
    ::Type{GPDecomp},
    y::AbstractVector{<:Real},
    t::AbstractVector{<:Real}=eachindex(y);
    period::Union{Real,Nothing}=nothing,
    bandwidth::Union{Real,Nothing}=nothing,
    kernel::Union{Kernel,NamedTuple}=_DEFAULT_KERNEL,
    approx_rate::Bool=false,
    θ_init::Union{Nothing,NamedTuple}=nothing,
    θ_fixed::AbstractVector{<:Union{Symbol,String}}=Symbol[],
    sig::AbstractVector{<:Real}=[0.5, 0.9],
    t_out::AbstractVector{<:Real}=t,
    optimizer=_default_optimizer,
    max_iter::Int=1_000,
)
    # Explicit kwargs take precedence; θ_init fields override the default (Inf).
    period_eff = if period !== nothing
        Float64(period)
    elseif (θ_init !== nothing && haskey(θ_init, :period))
        Float64(θ_init.period)
    else
        Inf
    end
    bandwidth_eff = if bandwidth !== nothing
        Float64(bandwidth)
    elseif (θ_init !== nothing && haskey(θ_init, :bandwidth))
        Float64(θ_init.bandwidth)
    else
        Inf
    end

    length(y) == length(t) || throw(DimensionMismatch("y and t must have the same length"))
    all(0 .< sig .< 1) || throw(DomainError(sig, "sig levels must be in (0, 1)"))
    isfinite(period_eff) &&
        period_eff <= 0 &&
        throw(DomainError(period_eff, "period must be positive"))
    isfinite(bandwidth_eff) &&
        bandwidth_eff <= 0 &&
        throw(DomainError(bandwidth_eff, "bandwidth must be positive"))
    isfinite(period_eff) &&
        bandwidth_eff < period_eff &&
        throw(
            DomainError(
                bandwidth_eff,
                "bandwidth must be at least the period ($period_eff): a shorter envelope " *
                "leaves no identifiable periodicity",
            ),
        )

    perm = sortperm(t)
    y_s = Float64.(y[perm])
    t_s = Float64.(t[perm])
    kernels = _resolve_kernels(kernel)

    result = _fit_gp(
        y_s,
        t_s;
        period=period_eff,
        bandwidth=bandwidth_eff,
        θ_init,
        θ_fixed=Symbol.(θ_fixed),
        kernels,
        optimizer,
        max_iter,
    )

    return _build_decomposition(
        result.posterior,
        Float64.(t_out),
        result.θ,
        result.ȳ,
        result.s_y;
        sig=Float64.(sig),
        approx_rate,
        fit_info=result.fit_info,
    )
end
