const _JITTER = 1e-6    # standardized units, training covariance only

const _default_optimizer = LBFGS(;
    alphaguess = Optim.LineSearches.InitialStatic(; scaled = true),
    linesearch = Optim.LineSearches.BackTracking()
)

function _optimize_params(θ₀::NamedTuple, fix::AbstractVector{Symbol}, lag::_LagIndex, y;
        kernels, optimizer, max_iter)
    x₀, unflatten, free = _flatten(_init_params(θ₀, fix))
    isempty(x₀) && return (θ = unflatten(x₀), free = free, result = nothing)
    v(x) = _lag_covariance(unflatten(x), lag.lags, kernels)
    function fg!(G, x)
        nll, w = _nll_lags(v(x), lag, y, !isnothing(G))
        isnothing(G) || mul!(G, ForwardDiff.jacobian(v, x)', w)
        return nll
    end
    objective = OnceDifferentiable(x -> fg!(nothing, x), (G, x) -> fg!(G, x), fg!, x₀)
    result = optimize(objective, x₀, optimizer, Optim.Options(; iterations = max_iter))
    return (θ = unflatten(Optim.minimizer(result)), free = free, result = result)
end

function _fit_gp(y::Vector{Float64}, t::Vector{Float64}; period, bandwidth,
        estimate_bandwidth, θ_init, θ_fixed, kernels, optimizer, max_iter)
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
    lag = _LagIndex(t)
    fit_kwargs = (; kernels, optimizer, max_iter)
    bandwidth_test = nothing
    if estimate_bandwidth && has_seasonal
        warm = haskey(θ_init, :bandwidth)
        opt, bandwidth_test = _fit_bandwidth(θ0, fix, t, lag, y_fit, warm; fit_kwargs...)
    else
        opt = _optimize_params(θ0, [fix; :bandwidth], lag, y_fit; fit_kwargs...)
    end
    θ_opt = opt.θ

    ks = _component_kernels(θ_opt, kernels)
    fx = GP(_signal_kernel(ks))(t, ks.noise + _JITTER)
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
        free, l_trend_min, at_bound, bandwidth_test, whiteness,
        loglik = to_original(logpdf(fx, y_fit)),
        optimizer = isnothing(res) ? nothing : Optim.summary(res),
        converged = isnothing(res) || Optim.converged(res),
        iterations = isnothing(res) ? 0 : Optim.iterations(res))
    return (; posterior = (; ks, data = post.data), θ = θ_final, θ_opt.period, ȳ, s_y,
        fit_info)
end

# 5% critical value of 2ΔlogL for the boundary null ρ = 0, the ½χ²₀ + ½χ²₁ mixture [1]
const _BANDWIDTH_LR_CRITICAL = quantile(Chisq(1), 0.90)
const _BANDWIDTH_MIN_CYCLES = 5    # fewer make the null distribution approximate
const _BANDWIDTH_START = 0.9       # ρ₀ ≈ one period, off the flat end of the logistic map

# Static fit A and free fits B (cold, ρ₀ = 0.9) and C (warm from A, ρ₀ at the
# log-bandwidth midpoint of B and the span); with a `warm` θ_init.bandwidth, A and one free
# fit from θ_init (from A at ρ₀ = period/span if that bandwidth is Inf); see
# `GaussianProcess`
function _fit_bandwidth(θ0, fix, t, lag, y, warm::Bool; kwargs...)
    P = θ0.period
    span = last(t) - first(t)
    nll(opt) = _nll(opt.θ, lag, y, kwargs[:kernels])
    function free_fit(θ)
        try
            return _optimize_params(θ, fix, lag, y; kwargs...)
        catch err
            err isa PosDefException || rethrow()
            return nothing
        end
    end

    static = _optimize_params(merge(θ0, (; ρ = 0.0)), [fix; :bandwidth], lag, y; kwargs...)
    nll_static = nll(static)
    fits = if warm
        [free_fit(θ0.ρ > 0 ? θ0 : merge(static.θ, (; ρ = min(P / span, _BANDWIDTH_START))))]
    else
        B = free_fit(merge(θ0, (; ρ = _BANDWIDTH_START)))
        ρ_B = isnothing(B) ? _BANDWIDTH_START : B.θ.ρ
        [
            B, free_fit(merge(static.θ, (;
                ρ = clamp(sqrt(ρ_B * P / span), 1e-4, _BANDWIDTH_START))))]
    end
    filter!(!isnothing, fits)
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

"""
    fit(model::GaussianProcess, y, t=eachindex(y); period=Inf) -> Decomposition

Decompose `y` observed at times `t` (in any order, possibly repeated) by maximum marginal
likelihood; `period` is the seasonal period in units of `t`, `Inf` for no seasonal
component. The components are at the sorted `t`, with 50% and 90% credible intervals;
`predict(d, t_new; sig)` gives other times or levels.
"""
function fit(model::GaussianProcess, y::AbstractVector{<:Real},
        t::AbstractVector{<:Real} = eachindex(y); period::Real = Inf)
    (; θ_init, θ_fixed) = model
    period > 0 || throw(DomainError(period, "period must be positive"))
    bandwidth = get(θ_fixed, :bandwidth, get(θ_init, :bandwidth, Inf))
    isfinite(period) && bandwidth < period &&
        throw(DomainError(bandwidth, "bandwidth must be at least the period ($period)"))
    y_s, t_s = _sorted_data(y, t)
    length(unique(y_s)) > 1 ||
        throw(ArgumentError("y must have at least two distinct values"))

    result = _fit_gp(y_s, t_s; period = Float64(period), bandwidth,
        estimate_bandwidth = !haskey(θ_fixed, :bandwidth),
        θ_init = merge(θ_init, θ_fixed), θ_fixed = collect(Symbol, keys(θ_fixed)),
        kernels = model.kernel, model.optimizer, model.max_iter)
    fit_info = (; result.posterior, result.ȳ, result.s_y, fit = result.fit_info)
    return _build_decomposition(model, fit_info, t_s, result.period, result.θ;
        sig = [0.5, 0.9])
end
