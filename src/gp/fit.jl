const _JITTER = 1e-6    # standardized units, training covariance only

const _default_optimizer = LBFGS(;
    alphaguess = Optim.LineSearches.InitialStatic(; scaled = true),
    linesearch = Optim.LineSearches.BackTracking()
)

function _finite_gp(θ, t, kernels)
    GP(_observation_kernel(_component_kernels(θ, kernels)))(t, _JITTER)
end

_nll(θ::NamedTuple, t, y, kernels) = -logpdf(_finite_gp(θ, t, kernels), y)

function _optimize_params(θ₀::NamedTuple, fix::AbstractVector{Symbol}, t, y;
        kernels, optimizer, autodiff, max_iter)
    wrapped = _init_params(θ₀, fix)
    x₀, unflatten = value_flatten(wrapped)
    free = [k for (k, v) in pairs(wrapped) if !(v isa ParameterHandling.Fixed)]
    isempty(x₀) && return (θ = unflatten(x₀), free = free, result = nothing)
    loss = x -> _nll(unflatten(x), t, y, kernels)
    result = optimize(loss, x₀, optimizer, Optim.Options(; iterations = max_iter); autodiff)
    return (θ = unflatten(Optim.minimizer(result)), free = free, result = result)
end

function _fit_gp(y::Vector{Float64}, t::Vector{Float64}; period, bandwidth,
        estimate_bandwidth, θ_init, θ_fixed, kernels, optimizer, autodiff, max_iter)
    ȳ, s_y = mean_and_std(y)
    y_fit = (y .- ȳ) ./ s_y

    θ0_base = _merge_params(θ_init, s_y)
    period = θ0_base.σ_seasonal == 0 ? Inf : Float64(period)
    has_seasonal = isfinite(period)
    l_trend_min = _trend_lower_bound(period)
    l_trend₀ = _initial_l_trend(θ0_base.l_trend, haskey(θ_init, :l_trend),
        :l_trend ∈ θ_fixed, l_trend_min)
    θ0 = (; θ0_base.σ_trend, l_trend = l_trend₀, θ0_base.σ_seasonal, θ0_base.l_seasonal,
        period, ρ = has_seasonal ? period / bandwidth : 0.0, θ0_base.σ_noise)

    fix = filter(!=(:bandwidth), θ_fixed)
    fit_kwargs = (; kernels, optimizer, autodiff, max_iter)
    bandwidth_test = nothing
    if estimate_bandwidth && has_seasonal
        opt, bandwidth_test = _fit_bandwidth(θ0, fix, t, y_fit; fit_kwargs...)
    else
        opt = _optimize_params(θ0, [fix; :bandwidth], t, y_fit; fit_kwargs...)
    end
    θ_opt = opt.θ

    ks = _component_kernels(θ_opt, kernels)
    fx = GP(_observation_kernel(ks))(t, _JITTER)
    post = posterior(fx, y_fit)

    at_bound = :l_trend ∈ opt.free && θ_opt.l_trend < (1 + 1e-3) * l_trend_min
    at_bound && @warn "l_trend is at its lower bound, the period $l_trend_min: variation " *
          "faster than the period is left to the seasonal and noise components"
    whiteness = _innovation_whiteness(post.data.C, y_fit, t, period)
    _warn_whiteness(whiteness, t, period)
    :l_seasonal ∈ opt.free && θ_opt.l_seasonal > (1 - 1e-3) * _L_SEASONAL_MAX &&
        @info "l_seasonal is at its cap $_L_SEASONAL_MAX: the seasonal pattern is " *
              "essentially a sinusoid"

    θ_final = _destandardize_params(θ_opt, s_y)
    isnothing(bandwidth_test) || _note_bandwidth(θ_final.bandwidth, θ_opt.ρ, t)
    to_original(ℓ) = ℓ - length(y) * log(s_y)    # Jacobian of the standardization
    if !isnothing(bandwidth_test)
        bandwidth_test = merge(bandwidth_test,
            (; loglik_static = to_original(bandwidth_test.loglik_static)))
    end
    free = replace(opt.free, :ρ => :bandwidth)
    isnothing(bandwidth_test) || :bandwidth ∈ free || push!(free, :bandwidth)
    res = opt.result
    fit_info = (;
        kernels, free, l_trend_min, at_bound, bandwidth_test, whiteness,
        loglik = to_original(logpdf(fx, y_fit)),
        optimizer = isnothing(res) ? nothing : Optim.summary(res),
        converged = isnothing(res) || Optim.converged(res),
        iterations = isnothing(res) ? 0 : Optim.iterations(res))
    return (; posterior = (; ks, data = post.data), θ = θ_final, ȳ, s_y, fit_info)
end

# 5% critical value of 2ΔlogL for the boundary null ρ = 0, the ½χ²₀ + ½χ²₁ mixture [1]
const _BANDWIDTH_LR_CRITICAL = quantile(Chisq(1), 0.90)
const _BANDWIDTH_MIN_CYCLES = 5    # fewer make the null distribution approximate
const _BANDWIDTH_START = 0.9       # ρ₀ ≈ one period, off the flat end of the logistic map

# Static fit A, free fits B (cold, ρ₀ = 0.9) and C (warm from A, ρ₀ at the log-bandwidth
# midpoint of B and the span), and D from a finite θ_init.bandwidth; see `fit(GPDecomp, ...)`
function _fit_bandwidth(θ0, fix, t, y; kwargs...)
    P = θ0.period
    span = last(t) - first(t)
    nll(opt) = _nll(opt.θ, t, y, kwargs[:kernels])
    function free_fit(θ)
        try
            return _optimize_params(θ, fix, t, y; kwargs...)
        catch err
            err isa PosDefException || rethrow()
            return nothing
        end
    end

    static = _optimize_params(merge(θ0, (; ρ = 0.0)), [fix; :bandwidth], t, y; kwargs...)
    nll_static = nll(static)
    B = free_fit(merge(θ0, (; ρ = _BANDWIDTH_START)))
    ρ_B = isnothing(B) ? _BANDWIDTH_START : B.θ.ρ
    C = free_fit(merge(static.θ, (;
        ρ = clamp(sqrt(ρ_B * P / span), 1e-4, _BANDWIDTH_START))))
    D = θ0.ρ > 0 ? free_fit(θ0) : nothing
    fits = filter(!isnothing, [B, C, D])
    isempty(fits) && return static, nothing
    nlls = nll.(fits)
    best, nll_best = fits[argmin(nlls)], minimum(nlls)

    lr = max(2 * (nll_static - nll_best), 0.0)
    static_selected = lr < _BANDWIDTH_LR_CRITICAL
    approximate = span / P < _BANDWIDTH_MIN_CYCLES ||
                  static.θ.l_seasonal > (1 - 1e-3) * _L_SEASONAL_MAX ||
                  static.θ.l_trend < (1 + 1e-3) * P
    test = (; lr_static = lr, critical = _BANDWIDTH_LR_CRITICAL, static_selected,
        loglik_static = -nll_static, approximate)
    return static_selected ? static : best, test
end

function _note_bandwidth(bandwidth::Real, ρ::Real, t)
    span = last(t) - first(t)
    isfinite(bandwidth) && bandwidth > span &&
        @info "bandwidth = $(_fmt(bandwidth)) exceeds the span $(_fmt(span)) and is " *
              "weakly determined (effectively a lower bound)"
    ρ ≥ 0.99 && @info "bandwidth is about one period: the seasonal pattern barely repeats"
    return nothing
end

# Typical sampling interval, robust to repeated times
_sampling_interval(t) = median(diff(unique(t)))

# Ljung–Box test of the standardized innovations L⁻¹y (K = LLᵀ), iid N(0, 1) under the model,
# over lags by index covering two seasonal cycles (ten without seasonality)
function _innovation_whiteness(C::Cholesky, y::AbstractVector{Float64}, t, period::Real)
    N = length(y)
    length(unique(t)) ≥ 2 || return nothing
    lags = isfinite(period) ? 2 * max(round(Int, period / _sampling_interval(t)), 1) : 10
    lags = min(lags, N ÷ 4)
    lags ≥ 1 || return nothing
    e = C.L \ y
    test = LjungBoxTest(e, lags)
    r = autocor(e, 1:lags)
    k = argmax(abs.(r))
    return (; test.Q, lags, pvalue = pvalue(test), max_lag = k, max_acf = r[k])
end

const _WHITENESS_LEVEL = 0.01

function _warn_whiteness(w, t, period::Real)
    (isnothing(w) || w.pvalue ≥ _WHITENESS_LEVEL) && return nothing
    p = isfinite(period) ? round(Int, period / _sampling_interval(t)) : 0
    hint = if p > 1 && w.max_lag % p == 0
        ", a seasonal lag: the seasonal pattern may evolve faster than the bandwidth allows"
    elseif w.max_acf > 0
        ": structure faster than the trend is left in the noise"
    else
        ""
    end
    @warn "the innovations are autocorrelated (Ljung–Box p = $(round(w.pvalue; sigdigits = 2)) " *
          "over lags 1:$(w.lags)); the largest autocorrelation, $(round(w.max_acf; digits = 2)), " *
          "is at lag $(w.max_lag)$hint. Lags are by index, assuming roughly regular sampling"
    return nothing
end

const _GP_θ_KEYS = (
    :σ_trend, :l_trend, :σ_seasonal, :l_seasonal, :period, :bandwidth, :σ_noise)

# Starting values (θ_fixed overriding θ_init, with a warning) and the names held fixed
function _merge_init_fixed(θ_init::NamedTuple, θ_fixed::NamedTuple)
    _check_keys(θ_init, _GP_θ_KEYS, "θ_init")
    _check_keys(θ_fixed, _GP_θ_KEYS, "θ_fixed")
    both = [k for k in keys(θ_init) if haskey(θ_fixed, k)]
    isempty(both) ||
        @warn "θ_init and θ_fixed both give $(join(both, ", ")): using the θ_fixed value"
    return merge(θ_init, θ_fixed), Symbol[k for k in keys(θ_fixed) if k != :period]
end

"""
    fit(GPDecomp, y, t=eachindex(y); kwargs...) -> GPDecomp

Decompose `y` observed at times `t` into trend, seasonal and noise Gaussian processes by
maximum marginal likelihood. Hyperparameters and components are in the units of `y`.

```julia
fit(GPDecomp, y, t; period=nothing, kernel=Matern52Kernel(), approx_rate=false,
    θ_init=(;), θ_fixed=(;), sig=[0.5, 0.9], t_out=t, optimizer=LBFGS(...),
    autodiff=AutoZygote(), max_iter=1000)
```

- `period`: seasonal period in units of `t`, never optimized; `Inf` for no seasonal
  component. Defaults to `θ_fixed.period` or `θ_init.period`, else `Inf`.
- `kernel`: a `Kernel` for every component, or a `NamedTuple` with keys `trend` and
  `seasonal`, the latter a `Kernel` or a `NamedTuple` with keys `effect` and `envelope`;
  unspecified components take `Matern52Kernel()`.
- `approx_rate`: skip the rate's posterior variance (`std` is `NaN`).
- `θ_init`: starting values of hyperparameters among `σ_trend`, `l_trend`, `σ_seasonal`,
  `l_seasonal`, `bandwidth`, `σ_noise` (e.g. the `θ` of a previous fit). `σ = 0` and
  `l_trend = Inf` are held fixed.
- `θ_fixed`: hyperparameters held fixed, with the same keys; they override `θ_init`.
- `sig`: credible interval levels.
- `t_out`: times at which the components are returned.
- `optimizer`, `max_iter`: Optim.jl optimizer and iteration limit.
- `autodiff`: ADTypes.jl backend for the gradient of the marginal likelihood.

## Identifiability

The seasonal kernel has its mean over a period removed, so the seasonal component cannot
absorb the trend, and `l_trend` is constrained above `period` (a fixed one must exceed it).
`l_seasonal` is capped at 10. A warning is emitted if `l_trend` ends on its bound, or if the
standardized innovations fail a Ljung–Box test at the 1% level (`d._fit_info.fit.whiteness`).

## Bandwidth estimation

The seasonal envelope has lengthscale `bandwidth ≥ period`, parametrized by
`ρ = period / bandwidth ∈ [0, 1]`; `bandwidth = Inf` is a static pattern. Unless it is fixed,
the static model is fitted together with free fits from `ρ = 0.9`, from the log-midpoint of
that fit's bandwidth and the span of `t` (warm-started from the static fit), and from a
finite `θ_init.bandwidth`. The best free fit is kept if `2(logL - logL_static)` exceeds the
5% critical value 2.71 of its `½χ²₀ + ½χ²₁` null distribution [1], else `bandwidth = Inf`.
The test is in `d._fit_info.fit.bandwidth_test`, flagged `approximate` below five periods
or when the static fit has `l_seasonal` or `l_trend` on a bound.

[1] S. G. Self and K.-Y. Liang. "Asymptotic properties of maximum likelihood estimators and
likelihood ratio tests under nonstandard conditions". JASA 82 (398), 1987.
"""
function fit(::Type{GPDecomp}, y::AbstractVector{<:Real},
        t::AbstractVector{<:Real} = eachindex(y);
        period::Union{Real, Nothing} = nothing,
        kernel::Union{Kernel, NamedTuple} = _DEFAULT_KERNEL,
        approx_rate::Bool = false,
        θ_init::NamedTuple = (;),
        θ_fixed::NamedTuple = (;),
        sig::AbstractVector{<:Real} = [0.5, 0.9],
        t_out::AbstractVector{<:Real} = t,
        optimizer = _default_optimizer,
        autodiff::AbstractADType = AutoZygote(),
        max_iter::Int = 1_000)
    θ_init, fixed_keys = _merge_init_fixed(θ_init, θ_fixed)
    period = Float64(something(period, get(θ_init, :period, Inf)))
    bandwidth = Float64(get(θ_init, :bandwidth, Inf))

    y_s, t_s = _sorted_data(y, t)
    length(unique(y_s)) > 1 ||
        throw(ArgumentError("y must have at least two distinct values"))
    all(0 .< sig .< 1) || throw(DomainError(sig, "sig levels must be in (0, 1)"))
    isfinite(period) && period <= 0 && throw(DomainError(period, "period must be positive"))
    bandwidth <= 0 && throw(DomainError(bandwidth, "bandwidth must be positive"))
    isfinite(period) && bandwidth < period &&
        throw(DomainError(bandwidth, "bandwidth must be at least the period ($period)"))

    result = _fit_gp(y_s, t_s; period, bandwidth,
        estimate_bandwidth = :bandwidth ∉ fixed_keys, θ_init, θ_fixed = fixed_keys,
        kernels = _resolve_kernels(kernel), optimizer, autodiff, max_iter)
    return _build_decomposition(result.posterior, Float64.(t_out), result.θ, result.ȳ,
        result.s_y; sig = Float64.(sig), approx_rate, fit_info = result.fit_info)
end
