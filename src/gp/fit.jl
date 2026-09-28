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
    estimate_bandwidth::Bool=false,
    θ_init::Union{Nothing,NamedTuple}=nothing,
    θ_fixed::AbstractVector{Symbol}=Symbol[],
    kernels::NamedTuple=_default_kernels,
    optimizer=_default_optimizer,
    max_iter::Int=1_000,
)
    y_fit, ȳ, s_y = standardize(y)
    y_fit = Float64.(y_fit)

    # Defaults are all-ones in standardized space. User θ_init is in original units;
    # _merge_params divides σ fields by s_y before merging.
    θ0_base = _merge_params(θ_init, s_y)
    period_eff = θ0_base.σ_seasonal == 0.0 ? Inf : Float64(period)
    has_seasonal = isfinite(period_eff)
    l_trend_min = _trend_lower_bound(period_eff)
    l_trend₀ = _initial_l_trend(
        θ0_base.l_trend,
        θ_init !== nothing && haskey(θ_init, :l_trend),
        :l_trend ∈ θ_fixed,
        l_trend_min,
    )
    θ0 = (;
        σ_trend=θ0_base.σ_trend,
        l_trend=l_trend₀,
        σ_seasonal=θ0_base.σ_seasonal,
        l_seasonal=θ0_base.l_seasonal,
        period=period_eff,
        ρ=has_seasonal ? period_eff / bandwidth : 0.0,
        σ_noise=θ0_base.σ_noise,
    )

    t_obs = GPPPInput(:y, collect(Float64, t))
    fix = filter(!=(:bandwidth), θ_fixed)
    fit_kwargs = (; kernels, optimizer, max_iter)
    bandwidth_test = nothing
    if estimate_bandwidth && has_seasonal
        opt, bandwidth_test = _fit_bandwidth(θ0, fix, t_obs, y_fit, t; fit_kwargs...)
    else
        opt = _optimize_params(θ0, [fix; :bandwidth], t_obs, y_fit; fit_kwargs...)
    end
    θ_opt = opt.θ

    gp = _build_gp(θ_opt, kernels)
    fx = gp(t_obs, _JITTER)
    posterior_gp = posterior(fx, y_fit)

    at_bound = :l_trend ∈ opt.free && θ_opt.l_trend < (1 + 1e-3) * l_trend_min
    at_bound && @warn "l_trend is at its lower bound, the period $l_trend_min: the " *
        "data favor a trend that varies faster than the seasonal period allows, and " *
        "that variation is left to the seasonal and noise components"
    whiteness = _innovation_whiteness(fx, y_fit, t, period_eff)
    _warn_whiteness(whiteness, t, period_eff)
    :l_seasonal ∈ opt.free &&
        θ_opt.l_seasonal > (1 - 1e-3) * _L_SEASONAL_MAX &&
        @info "l_seasonal is at its cap $_L_SEASONAL_MAX: the seasonal pattern is " *
            "essentially a pure sinusoid"

    θ_final = _destandardize_params(θ_opt, s_y)
    isnothing(bandwidth_test) || _note_bandwidth(θ_final.bandwidth, θ_opt.ρ, t)
    # log marginal likelihoods of the original (unstandardized) data
    to_original(ℓ) = ℓ - length(y) * log(s_y)
    if !isnothing(bandwidth_test)
        bandwidth_test = merge(
            bandwidth_test, (; loglik_static=to_original(bandwidth_test.loglik_static))
        )
    end
    free = replace(opt.free, :ρ => :bandwidth)
    isnothing(bandwidth_test) || :bandwidth ∈ free || push!(free, :bandwidth)
    fit_info = (;
        kernels,
        free,
        l_trend_min,
        at_bound,
        bandwidth_test,
        whiteness,
        loglik=to_original(logpdf(fx, y_fit)),
        optimizer=isnothing(opt.result) ? nothing : Optim.summary(opt.result),
        converged=isnothing(opt.result) || Optim.converged(opt.result),
        iterations=isnothing(opt.result) ? 0 : Optim.iterations(opt.result),
    )

    return (posterior=posterior_gp, θ=θ_final, ȳ=ȳ, s_y=s_y, fit_info=fit_info)
end

# 5% critical value of the likelihood-ratio statistic 2(logL_free - logL_static) for the
# boundary null ρ = 0: the ½χ²₀ + ½χ²₁ mixture of Self & Liang (1987)
const _BANDWIDTH_LR_CRITICAL = quantile(Chisq(1), 0.90)

# Fewer cycles than this make the null distribution of the likelihood ratio approximate
const _BANDWIDTH_MIN_CYCLES = 5

# Estimate the bandwidth by three fits (for ρ = P / bandwidth and the span T):
#   A. the static model, ρ = 0 fixed ("from infinity"): the null of the test;
#   B. free ρ from ρ₀ = 0.9 (about one period), nuisances cold from θ0;
#   C. free ρ from the log-bandwidth midpoint of B's and A's end points, with T standing
#      in for A's ∞, i.e. ρ₀ = √(ρ̂_B P / T), nuisances warm from A.
# B and C cover each other's traps: warm starts from a static fit with a nuisance at a
# bound (e.g. l_seasonal at its cap) all stay near static, while a cold start can end in
# a worse local optimum. ρ₀ = 0.9 rather than 1 - ε, where the logistic map is flat. The
# better of B and C is kept over A only if the likelihood ratio exceeds the boundary
# critical value. A finite θ_init.bandwidth adds a free fit from θ0. Returns the chosen
# fit and the test.
function _fit_bandwidth(θ0, fix, t_obs, y, t; kwargs...)
    P = θ0.period
    span = last(t) - first(t)
    nll(opt) = _nll(opt.θ, t_obs, y, kwargs[:kernels])
    function free_fit(θ)
        try
            return _optimize_params(θ, fix, t_obs, y; kwargs...)
        catch err
            err isa PosDefException || rethrow()
            return nothing
        end
    end

    static =
        _optimize_params(merge(θ0, (; ρ=0.0)), [fix; :bandwidth], t_obs, y; kwargs...)
    nll_static = nll(static)
    B = free_fit(merge(θ0, (; ρ=_BANDWIDTH_START_SMALL)))
    ρ_B = isnothing(B) ? _BANDWIDTH_START_SMALL : B.θ.ρ
    ρ_C = clamp(sqrt(ρ_B * P / span), 1e-4, _BANDWIDTH_START_SMALL)
    C = free_fit(merge(static.θ, (; ρ=ρ_C)))
    D = θ0.ρ > 0 ? free_fit(θ0) : nothing    # from θ_init.bandwidth
    fits = filter(!isnothing, [B, C, D])
    isempty(fits) && return static, nothing
    nlls = nll.(fits)
    best, nll_best = fits[argmin(nlls)], minimum(nlls)

    lr = max(2 * (nll_static - nll_best), 0.0)
    static_selected = lr < _BANDWIDTH_LR_CRITICAL
    approximate =
        span / P < _BANDWIDTH_MIN_CYCLES ||
        static.θ.l_seasonal > (1 - 1e-3) * _L_SEASONAL_MAX ||
        static.θ.l_trend < (1 + 1e-3) * P
    test = (;
        lr_static=lr,
        critical=_BANDWIDTH_LR_CRITICAL,
        static_selected,
        loglik_static=-nll_static,
        approximate,
    )
    return static_selected ? static : best, test
end

const _BANDWIDTH_START_SMALL = 0.9

function _note_bandwidth(bandwidth::Real, ρ::Real, t)
    span = last(t) - first(t)
    isfinite(bandwidth) && bandwidth > span && @info "bandwidth = $(_fmt(bandwidth)) " *
        "exceeds the span $(_fmt(span)): the seasonal pattern drifts over the whole " *
        "record, and the bandwidth is weakly determined (effectively a lower bound)"
    ρ ≥ 0.99 && @info "bandwidth is about one period: the seasonal pattern barely " *
        "repeats from one cycle to the next"
    return nothing
end

# Ljung–Box test of the standardized innovations e = L⁻¹y (K = LLᵀ, the training
# covariance), which are iid N(0, 1) under the fitted model: in time order, its
# one-step-ahead prediction errors. Unlike the posterior-mean noise, whose covariance
# σ⁴K⁻¹ is colored by the smoother, they are white when the model is right. The lags
# are by index (regular sampling), covering two cycles of a seasonal component.
function _innovation_whiteness(fx, y::AbstractVector{Float64}, t, period::Real)
    N = length(y)
    Δ = median(diff(t))
    lags = isfinite(period) ? 2 * max(round(Int, period / Δ), 1) : 10
    lags = min(lags, N ÷ 4)
    lags ≥ 1 || return nothing
    e = cholesky(Symmetric(cov(fx))).L \ y
    e = e .- mean(e)
    r = [dot(view(e, 1:(N - k)), view(e, (k + 1):N)) for k in 1:lags] ./ sum(abs2, e)
    Q = N * (N + 2) * sum(r[k]^2 / (N - k) for k in 1:lags)
    k = argmax(abs.(r))
    return (; Q, lags, pvalue=ccdf(Chisq(lags), Q), max_lag=k, max_acf=r[k])
end

const _WHITENESS_LEVEL = 0.01

function _warn_whiteness(w, t, period::Real)
    (isnothing(w) || w.pvalue ≥ _WHITENESS_LEVEL) && return nothing
    p = isfinite(period) ? round(Int, period / median(diff(t))) : 0
    hint = if p > 1 && w.max_lag % p == 0
        ", a seasonal lag: the seasonal pattern may evolve faster than the bandwidth allows"
    elseif w.max_acf > 0
        ": structure faster than the trend is left in the noise"
    else
        ""
    end
    @warn "the innovations are autocorrelated (Ljung–Box p = " *
        "$(round(w.pvalue; sigdigits=2)) over lags 1:$(w.lags)); the largest " *
        "autocorrelation, $(round(w.max_acf; digits=2)), is at lag $(w.max_lag)$hint. " *
        "The lags are by index, so this assumes roughly regular sampling"
    return nothing
end

const _GP_θ_KEYS = (
    :σ_trend, :l_trend, :σ_seasonal, :l_seasonal, :period, :bandwidth, :σ_noise
)

# Starting values of all hyperparameters given (θ_fixed overriding θ_init, with a
# warning) and the names of those held fixed. The period is always fixed.
function _merge_init_fixed(θ_init::NamedTuple, θ_fixed::NamedTuple)
    for (name, θ) in (("θ_init", θ_init), ("θ_fixed", θ_fixed)), k in keys(θ)
        k ∈ _GP_θ_KEYS ||
            throw(ArgumentError("Unknown $name key :$k. Valid keys: $_GP_θ_KEYS"))
    end
    both = [k for k in keys(θ_init) if haskey(θ_fixed, k)]
    isempty(both) || @warn "θ_init and θ_fixed both give $(join(both, ", ")): fixing " *
        "$(length(both) == 1 ? "it" : "them") at the θ_fixed value"
    return merge(θ_init, θ_fixed), Symbol[k for k in keys(θ_fixed) if k != :period]
end

"""
    fit(GPDecomp, y, t; kwargs...) -> GPDecomp

Decompose time series `y` observed at times `t` into trend, seasonal, and noise
components using a Gaussian process model.

`y` is internally z-score standardized before fitting; all returned parameters
and component estimates are in the original scale.

# Keyword arguments
- `period`: seasonal period in the same units as `t`, never optimized. `Inf` (default)
  disables the seasonal component. A `period` in `θ_fixed` or `θ_init` (e.g. from a
  fitted `θ`) is used as the default; the `period` keyword overrides it.
- `kernel`: base kernel(s) of the components (default `Matern52Kernel()`). A single
  kernel applies to every component. A `NamedTuple` with keys `trend` and `seasonal`
  sets them separately, and `seasonal` may itself be a kernel (for both the periodic
  effect and its envelope) or a `NamedTuple` with keys `effect` and `envelope`.
  Unspecified components default to `Matern52Kernel()`. Examples:
  `kernel=SEKernel()`, `kernel=(trend=SEKernel(), seasonal=Matern32Kernel())`,
  `kernel=(seasonal=(effect=Matern32Kernel(), envelope=SEKernel()),)`.
- `approx_rate`: use fast finite-difference rate with `std=NaN` (default `false`);
  set `true` to skip the posterior covariance computation for speed
- `θ_init`: `NamedTuple` of starting values for hyperparameters to be fitted, among
  `(σ_trend, l_trend, σ_seasonal, l_seasonal, bandwidth, σ_noise)`; the others start
  from defaults. Boundary values (`σ=0`, `l_trend=Inf`) are automatically fixed. The
  `θ` of a fitted decomposition can be passed back. `bandwidth` is the lengthscale of
  the seasonal envelope, over which the seasonal pattern evolves (at least `period`);
  `Inf` gives a static (purely periodic) pattern. It is estimated with the other
  hyperparameters, including whether the pattern is static (see *Bandwidth
  estimation*); a finite `θ_init.bandwidth` adds a start from it.
- `θ_fixed`: `NamedTuple` of values of hyperparameters held fixed, with the same keys,
  e.g. `θ_fixed=(bandwidth=36,)` for a fixed envelope, `θ_fixed=(bandwidth=Inf,)` for
  a static pattern or `θ_fixed=(σ_noise=0.1,)`. A hyperparameter in both `θ_init` and
  `θ_fixed` is fixed at the `θ_fixed` value, with a warning.
- `sig`: credible interval levels, e.g. `[0.5, 0.9]` (default)
- `t_out`: output time grid; defaults to `t`
- `optimizer`: Optim optimizer (default L-BFGS with backtracking)
- `max_iter`: maximum optimizer iterations (default 1000)

# Identifiability
The seasonal kernel has its mean over one period removed, so the seasonal component
averages to zero over each period and cannot absorb the trend; with an envelope, the
seasonal spectrum is concentrated around the harmonics `k / period`, `k ≠ 0`, and leaks
little to low frequencies. The trend and seasonal components are then separated by
`l_trend > period > seasonal timescale > 0 (noise)`: `l_trend` is optimized strictly
above `period` (unbounded without seasonality), and a fixed `l_trend` must exceed it,
with a warning if the optimum lies on the bound. The bandwidth is not ordered against
`l_trend`: a seasonal pattern may evolve faster or slower than the trend. `l_seasonal`
is capped at 10, beyond which the seasonal kernel is a pure sinusoid.

A warning is also emitted if the standardized innovations `L⁻¹y` (`K = LLᵀ` the
training covariance), which are white noise under the fitted model, fail a Ljung–Box
test at the 1% level over lags covering two seasonal cycles (ten without seasonality),
reporting the lag of the largest autocorrelation. The lags are by index, so this
assumes roughly regular sampling.

# Bandwidth estimation
The envelope is parametrized by `ρ = period / bandwidth ∈ [0, 1]`, whose boundary
`ρ = 0` is the static model. The likelihood is flat in `ρ` there, so it cannot be
reached by optimization. Three fits are made instead: (A) the static model itself,
"from infinity"; (B) a free fit from about one period (`ρ = 0.9`), with the other
hyperparameters from `θ_init` or their defaults; and (C) a free fit from the midpoint
in log-bandwidth of the end points of B and A, taking the span `T` of `t` for A's ∞
(`ρ = √(ρ̂_B · period / T)`), with the other hyperparameters from A. A finite
`θ_init.bandwidth` adds a fourth free fit from it, with the other hyperparameters as in
B. The best free fit is kept only if the likelihood ratio `2(logL - logL_static)`
exceeds 2.71, the 5% critical value of its `½χ²₀ + ½χ²₁` null distribution; otherwise
`bandwidth = Inf`. The test is in `d._fit_info.fit.bandwidth_test`, flagged
`approximate` when the series spans fewer than five periods or the static fit has
`l_seasonal` or `l_trend` at a bound. A bandwidth longer than the span is kept, with a
note that it is weakly determined. Estimation costs three fits (four with
`θ_init.bandwidth`).
"""
function fit(
    ::Type{GPDecomp},
    y::AbstractVector{<:Real},
    t::AbstractVector{<:Real}=eachindex(y);
    period::Union{Real,Nothing}=nothing,
    kernel::Union{Kernel,NamedTuple}=_DEFAULT_KERNEL,
    approx_rate::Bool=false,
    θ_init::NamedTuple=(;),
    θ_fixed::NamedTuple=(;),
    sig::AbstractVector{<:Real}=[0.5, 0.9],
    t_out::AbstractVector{<:Real}=t,
    optimizer=_default_optimizer,
    max_iter::Int=1_000,
)
    θ_init, fixed = _merge_init_fixed(θ_init, θ_fixed)
    # The explicit kwarg takes precedence over θ_fixed and θ_init, then the default (Inf)
    period_eff = if period !== nothing
        Float64(period)
    elseif haskey(θ_init, :period)
        Float64(θ_init.period)
    else
        Inf
    end
    # The bandwidth is estimated, from θ_init.bandwidth as an extra start, unless fixed
    estimate_bandwidth = :bandwidth ∉ fixed
    bandwidth_eff = haskey(θ_init, :bandwidth) ? Float64(θ_init.bandwidth) : Inf

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
        estimate_bandwidth,
        θ_init,
        θ_fixed=fixed,
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
