"""
    STLDecomp <: AbstractDecomp

Seasonal-trend decomposition with a loess trend. The fitted filters are in
`d.θ.filters`. Point estimates only (`std` is `NaN`). See `fit(STLDecomp, ...)`.

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
- `trend_window`: odd number of observations spanned by the local-quadratic loess trend
  (STL uses local-linear). Defaults to Cleveland et al.'s rule
  `n_t ≥ 1.5 n_p / (1 - 1.5 / n_s)` with the smallest seasonal window `n_s = 7`.

The rate is the derivative of the trend per unit of `t`.
"""
function fit(
    ::Type{STLDecomp},
    y::AbstractVector{<:Real},
    t::AbstractVector{<:Real}=eachindex(y);
    period::Real,
    trend_window::Union{Nothing,Integer}=nothing,
)
    y, t, Δt = _uniform_grid(y, t)
    T = _samples_per_period(period, Δt)
    trend_window = something(trend_window, _nextodd(1.5T / (1 - 1.5 / 7)))
    _halfwidth(trend_window, "trend_window")
    length(y) ≥ max(trend_window, 2T) ||
        throw(ArgumentError("need at least max(trend_window, two periods) observations"))

    initial_trend = fit(MovingAverage, y; window=T)
    detrended = y - predict(initial_trend, y)
    seasonal_filter = fit(StableSeasonalFilter, detrended; period=T)
    seasonal = predict(seasonal_filter, detrended)
    adjusted = y - seasonal
    trend_filter = fit(LoessFilter, adjusted; window=trend_window)
    trend = predict(trend_filter, adjusted)

    filters = (; initial_trend, seasonal=seasonal_filter, trend=trend_filter)
    return STLDecomp(
        t,
        Component(trend),
        Component(seasonal),
        Component(adjusted - trend),
        Component(_trend_rate(trend, Δt)),
        (; period=Float64(period), Δt, filters),
    )
end
