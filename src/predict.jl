function _component_estimate(
    post::PosteriorGP,
    key::Symbol,
    t::AbstractVector{Float64},
    sig::AbstractVector{Float64},
    ȳ::Float64,
    s::Float64,
    logscale::Bool,
    add_mean::Bool,
)
    jitter = 1e-12
    fg = post(GPPPInput(key, t), jitter)
    μ_std = mean(fg)
    # Derivative covariance (key == :r) accumulates floating-point error that
    # can make the posterior Schur complement slightly negative; clamp to zero.
    σ_std = sqrt.(max.(0.0, var(fg)))

    μ = s .* μ_std .+ (add_mean ? ȳ : 0.0)
    σ = s .* σ_std

    z = map(sig) do p
        quantile(Normal(), (1 + p) / 2)
    end
    lower = [μ .- ζ .* σ for ζ ∈ z]
    upper = [μ .+ ζ .* σ for ζ ∈ z]

    if logscale
        delog = add_mean ? expm1 : exp
        μ     = delog.(μ)
        lower = [delog.(l) for l ∈ lower]
        upper = [delog.(u) for u ∈ upper]
    end

    return ComponentEstimate(μ, σ, lower, upper, sig)
end

function _build_decomposition(
    post::PosteriorGP,
    t_out::AbstractVector{Float64},
    params::NamedTuple,
    ȳ::Float64,
    s::Float64;
    sig::AbstractVector{Float64},
    logscale::Bool,
    rate::Bool,
)
    trend    = _component_estimate(post, :x̄, t_out, sig, ȳ, s, logscale, true)
    seasonal = _component_estimate(post, :x̃, t_out, sig, 0.0, s, logscale, false)
    noise    = _component_estimate(post, :ϵ,  t_out, sig, 0.0, s, logscale, false)
    r = if rate
        _component_estimate(post, :r, t_out, sig, 0.0, s, false, false)
    else
        nothing
    end

    fit_info = (ȳ=ȳ, s=s, logscale=logscale)
    return Decomposition(t_out, trend, seasonal, noise, r, params, post, fit_info)
end

"""
    decompose(y, t; period, bandwidth=Inf, kwargs...) -> Decomposition

Decompose time series `y` observed at times `t` into trend, seasonal, and noise
components using a Gaussian process model.

# Arguments
- `y`: observation vector
- `t`: time vector (sorted or unsorted; need not be uniform)

# Keyword arguments
- `period`: seasonal period (required)
- `bandwidth`: temporal envelope lengthscale for quasi-periodic seasonality.
  `Inf` (default) gives a purely stationary periodic seasonal component.
  A finite value lets the seasonal pattern evolve smoothly over time.
- `logscale`: set `true` for multiplicative (log-scale) decomposition
- `standardize`: standardize `y` before fitting and back-transform outputs (default `true`)
- `rate`: also estimate the trend rate of change `r = dx̄/dt` (default `false`)
- `init_params`: `NamedTuple` of initial hyperparameter values
- `fix_params`: `Vector{Symbol}` of hyperparameters to hold fixed during optimization
- `sig`: credible interval levels, e.g. `[0.5, 0.9]` (default)
- `t_out`: output time grid; defaults to `t`
- `optimizer`: Optim optimizer (default L-BFGS with backtracking)
- `max_iter`: maximum optimizer iterations (default 1000)
"""
function decompose(
    y::AbstractVector{<:Real},
    t::AbstractVector{<:Real}=eachindex(y);
    period::Real,
    bandwidth::Real=Inf,
    logscale::Bool=false,
    standardize::Bool=true,
    rate::Bool=false,
    init_params::Union{Nothing, NamedTuple}=nothing,
    fix_params::AbstractVector{<:Union{Symbol,String}}=Symbol[],
    sig::AbstractVector{<:Real}=[0.5, 0.9],
    t_out::AbstractVector{<:Real}=t,
    optimizer=_default_optimizer,
    max_iter::Int=1_000,
)
    @assert length(y) == length(t) "y and t must have the same length"
    @assert period > 0 "period must be positive"
    @assert all(0 .< sig .< 1) "sig levels must be in (0, 1)"

    perm = sortperm(t)
    y_s = logscale ? log1p.(y[perm]) : Float64.(y[perm])
    t_s = Float64.(t[perm])

    fix = unique(Symbol.(fix_params))
    # Always seed period and bandwidth from the user's kwargs so the optimizer
    # starts at a meaningful point even when init_params is nothing.
    base = _merge_params(init_params)
    init = (base..., period=Float64(period), bandwidth=Float64(bandwidth))

    result = _fit_gp(y_s, t_s;
        standardize_data=standardize,
        init_params=init,
        fix_params=fix,
        rate=rate,
        bandwidth=Float64(bandwidth),
        optimizer=optimizer,
        max_iter=max_iter,
    )

    t_out_f = Float64.(t_out)
    return _build_decomposition(
        result.posterior, t_out_f, result.params, result.ȳ, result.s;
        sig=Float64.(sig), logscale=logscale, rate=rate,
    )
end

"""
    predict(d::Decomposition, t_new; sig=d.trend.sig) -> Decomposition

Re-evaluate a fitted decomposition at new time points without re-fitting.
"""
function predict(
    d::Decomposition,
    t_new::AbstractVector{<:Real};
    sig::AbstractVector{<:Real}=d.trend.sig,
)
    t_out = Float64.(t_new)
    fi = d._fit_info
    return _build_decomposition(
        d._posterior, t_out, d.params, fi.ȳ, fi.s;
        sig=Float64.(sig), logscale=fi.logscale, rate=!isnothing(d.rate),
    )
end
