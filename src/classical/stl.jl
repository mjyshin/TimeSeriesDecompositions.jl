"""
    STLDecomp <: AbstractDecomp

Seasonal-trend decomposition with a loess trend. The fitted filters are in
`d.θ.filters` and the hyperparameters chosen from the data in `d.θ.estimated`. Point
estimates only (`std` is `NaN`). See `fit(STLDecomp, ...)`.

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
    fit(STLDecomp, y, t=eachindex(y); period, trend_window) -> STLDecomp

Loess-based seasonal-trend decomposition of `y` observed at uniformly spaced times `t`.
**Placeholder** for STL; see [`STLDecomp`](@ref).

# Keyword arguments
- `period`: seasonal period in units of `t` (required); `period / Δt` must be an integer.
- `trend_window`: number of nearest observations in the local-linear loess trend.
  Defaults to Cleveland et al.'s rule `n_t ≥ 1.5 n_p / (1 - 1.5 / n_s)` with the smallest
  seasonal window `n_s = 7`.

Hyperparameters chosen from the data are reported with an `@info` message and stored in
`d.θ.estimated`. The rate is the derivative of the trend per unit of `t`.
"""
function fit(
    ::Type{STLDecomp},
    y::AbstractVector{<:Real},
    t::AbstractVector{<:Real}=eachindex(y);
    period::Real,
    trend_window::Union{Nothing,Integer}=nothing,
)
    y, t, Δt = _uniform_grid(y, t)
    p = _samples_per_period(period, Δt)
    window = something(trend_window, _nextodd(1.5p / (1 - 1.5 / 7)))
    estimated = isnothing(trend_window) ? (; trend_window=window) : (;)
    _halfwidth(window, "trend_window")
    length(y) ≥ max(window, 2p) ||
        throw(ArgumentError("need at least max(trend_window, two periods) observations"))

    initial_trend = fit(MovingAverage, y; window=p)
    detrended = y - predict(initial_trend, y)
    seasonal_filter = fit(StableSeasonalFilter, detrended; period=p)
    seasonal = predict(seasonal_filter, detrended)
    adjusted = y - seasonal
    trend_filter = _fit(LoessSmoother, adjusted, t; span=window / length(y), degree=1)
    trend = predict(trend_filter)

    _log_estimated(STLDecomp, estimated)
    filters = (; initial_trend, seasonal=seasonal_filter, trend=trend_filter)
    return STLDecomp(
        t,
        Component(trend),
        Component(seasonal),
        Component(adjusted - trend),
        Component(_trend_rate(trend, Δt)),
        (; period=Float64(period), Δt, filters, estimated),
    )
end
