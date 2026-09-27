"""
    X11Decomp <: AbstractDecomp

X-11-style seasonal-trend decomposition from classical filters: Henderson trend filters
with asymmetric end weights and `S(n × m)` seasonal filters. The fitted filters are in
`d.θ.filters`. Point estimates only (`std` is `NaN`). See `fit(X11Decomp, ...)`.
"""
struct X11Decomp <: AbstractDecomp
    t::Vector{Float64}
    trend::Component
    seasonal::Component
    noise::Component
    rate::Component
    θ::NamedTuple
end

"""
    fit(X11Decomp, y, t=eachindex(y); period, kwargs...) -> X11Decomp

X-11-style decomposition of `y` observed at uniformly spaced times `t`, in two passes:

 1. a centred moving average of one period gives an initial trend; an `S(n₁ × m₁)`
    seasonal filter of the detrended series gives a preliminary seasonal; a Henderson
    filter of the seasonally adjusted series gives a preliminary trend;
 2. an `S(n₂ × m₂)` seasonal filter of the series detrended by the preliminary trend gives
    the seasonal; a Henderson filter of the seasonally adjusted series gives the trend.

The noise is the remainder, `y - trend - seasonal`.

# Keyword arguments
- `period`: seasonal period in units of `t` (required); `period / Δt` must be an integer.
- `trend_window`: odd length of the final Henderson filter in samples (default `13`).
- `endpoints`: end weights of the final Henderson filter, an [`AbstractEndpoints`](@ref)
  or `nothing` for `NaN` ends. Defaults to X-11's Musgrave weights,
  `musgrave(musgrave_ic_ratio(trend_window))`.
- `preliminary_trend_window`, `preliminary_endpoints`: as above for the first pass
  (defaults `trend_window` and its Musgrave weights).
- `seasonal_filters`: `((n₁, m₁), (n₂, m₂))` (default `((3, 3), (3, 5))`, i.e. S3×3 then
  S3×5).

Unlike X-13ARIMA-SEATS, the filter lengths are not chosen from the data, the seasonal is
not centred, and there is no extreme-value adjustment.

The rate is the derivative of the trend per unit of `t`.
"""
function fit(
    ::Type{X11Decomp},
    y::AbstractVector{<:Real},
    t::AbstractVector{<:Real}=eachindex(y);
    period::Real,
    trend_window::Integer=13,
    endpoints::Union{Nothing,AbstractEndpoints}=musgrave(musgrave_ic_ratio(trend_window)),
    preliminary_trend_window::Integer=trend_window,
    preliminary_endpoints::AbstractEndpoints=musgrave(
        musgrave_ic_ratio(preliminary_trend_window)
    ),
    seasonal_filters::NTuple{2,NTuple{2,Integer}}=((3, 3), (3, 5)),
)
    y, t, Δt = _uniform_grid(y, t)
    T = _samples_per_period(period, Δt)
    length(y) ≥ 2T || throw(ArgumentError("need at least two periods of data"))
    (n₁, m₁), (n₂, m₂) = seasonal_filters

    initial_trend = fit(MovingAverage, y; window=T)
    detrended = y - predict(initial_trend, y)
    preliminary_seasonal = fit(SeasonalMovingAverage, detrended; period=T, n=n₁, m=m₁)
    adjusted = y - predict(preliminary_seasonal, detrended)
    preliminary_trend = fit(
        LocalPolynomialFilter,
        adjusted;
        window=preliminary_trend_window,
        endpoints=preliminary_endpoints,
    )

    detrended = y - predict(preliminary_trend, adjusted)
    seasonal_filter = fit(SeasonalMovingAverage, detrended; period=T, n=n₂, m=m₂)
    seasonal = predict(seasonal_filter, detrended)
    adjusted = y - seasonal
    trend_filter = fit(LocalPolynomialFilter, adjusted; window=trend_window, endpoints)
    trend = predict(trend_filter, adjusted)

    filters = (;
        initial_trend,
        preliminary_seasonal,
        preliminary_trend,
        seasonal=seasonal_filter,
        trend=trend_filter,
    )
    return X11Decomp(
        t,
        Component(trend),
        Component(seasonal),
        Component(adjusted - trend),
        Component(_trend_rate(trend, Δt)),
        (; period=Float64(period), Δt, filters),
    )
end
