"""
    STLDecomp <: AbstractDecomp

Seasonal-trend decomposition with a loess trend. The fitted smoothers are in
`d.θ.smoothers` and the hyperparameters chosen from the data in `d.θ.estimated`;
`d.θ.grid` keeps the components at the observed times, from which `predict(d, t_new)`
interpolates. Point estimates only (`std` is `NaN`). See `fit(STLDecomp, ...)`.

!!! warning "Placeholder"
    This is **not yet** the STL procedure of Cleveland et al. (1990): there is no
    cycle-subseries smoothing, low-pass filter, inner/outer loop or robustness
    weighting. It removes a moving-average trend, estimates a *fixed* seasonal pattern
    (stable seasonal filter) and smooths the seasonally adjusted series with loess.
"""
struct STLDecomp <: AbstractDecomp
    t::Vector{Float64}
    trend::Component
    seasonal::Component
    noise::Component
    rate::Component
    θ::NamedTuple
end

"""
    fit(STLDecomp, y, t=eachindex(y); period, trend=LOESS(; degree=1)) -> STLDecomp

Loess-based seasonal-trend decomposition of `y` observed at uniformly spaced times `t`.
**Placeholder** for STL; see [`STLDecomp`](@ref).

# Keyword arguments
- `period`: seasonal period in units of `t` (required); `period / Δt` must be an integer.
- `trend`: an [`AbstractTrendSmoother`](@ref) (default local-linear [`LOESS`](@ref)). A
  loess span left as `nothing` covers Cleveland et al.'s `n_t` observations, the
  smallest odd integer `≥ 1.5 n_p / (1 - 1.5 / n_s)` with the smallest seasonal window
  `n_s = 7`.

Hyperparameters chosen from the data are reported with an `@info` message and stored in
`d.θ.estimated`. The rate is the derivative, per unit of `t`, of the not-a-knot cubic
spline through the trend.
"""
function fit(
    ::Type{STLDecomp},
    y::AbstractVector{<:Real},
    t::AbstractVector{<:Real}=eachindex(y);
    period::Real,
    trend::AbstractTrendSmoother=LOESS(; degree=1),
)
    y, t, Δt = _uniform_grid(y, t)
    p = _samples_per_period(period, Δt)
    N = length(y)
    N ≥ 2p || throw(ArgumentError("need at least two periods of observations"))
    trend, estimated = _resolve_stl_trend(trend, p, Δt, N)

    initial_trend = _fit(MovingAverage(; width=p * Δt), y, t)
    seasonal_fit, seasonal, _ = _seasonal(
        StableSeasonal(), y - predict(initial_trend), t, p, Δt
    )
    adjusted = y - seasonal
    trend_fit = _fit(trend, adjusted, t)
    trend_values = predict(trend_fit)
    estimated = (; estimated..., trend_fit.estimated...)

    _log_estimated(STLDecomp, estimated)
    smoothers = (; initial_trend, seasonal=seasonal_fit, trend=trend_fit)
    noise = adjusted - trend_values
    rate = _trend_rate(trend_values, Δt)
    grid = (; t, trend=trend_values, seasonal=seasonal, noise, rate)
    return STLDecomp(
        t,
        Component(trend_values),
        Component(seasonal),
        Component(noise),
        Component(rate),
        (; period=Float64(period), Δt, smoothers, estimated, grid),
    )
end

# Cleveland et al.'s trend window n_t for a loess span left as nothing; other smoothers
# as in X11Decomp
function _resolve_stl_trend(m::LOESS, p, _, N)
    isnothing(m.span) && isnothing(m.bandwidth) || return m, (;)
    q = _nextodd(1.5p / (1 - 1.5 / 7))
    span = q / N
    return LOESS(span, nothing, m.degree), (; trend_span=span, trend_neighbours=q)
end

_resolve_stl_trend(m::AbstractTrendSmoother, p, Δt, N) = _resolve_trend(m, p, Δt, N)
