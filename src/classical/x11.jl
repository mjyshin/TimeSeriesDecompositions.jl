"""
    X11Decomp <: AbstractDecomp

X-11-style seasonal-trend decomposition from classical filters: Henderson (or loess)
trend filters with minimum-revision end weights and `S(n × m)` seasonal filters, for any
integer period. The fitted filters are in `d.θ.filters` and the hyperparameters chosen from the
data in `d.θ.estimated`. Point estimates only (`std` is `NaN`). See
`fit(X11Decomp, ...)`.
"""
struct X11Decomp <: AbstractDecomp
    t::Vector{Float64}
    trend::Component
    seasonal::Component
    noise::Component
    rate::Component
    θ::NamedTuple
end

const _SeasonalFilterSpec = Union{Nothing,Tuple{Integer,Integer},Type{StableSeasonalFilter}}

"""
    fit(X11Decomp, y, t=eachindex(y); period, kwargs...) -> X11Decomp

X-11-style decomposition of `y` observed at uniformly spaced times `t`, with `p` samples
per seasonal cycle, in two passes:

 1. a centred `2 × p` (even `p`) or `p`-term (odd `p`) moving average gives an initial
    trend, undefined in the first and last `p/2` points; a seasonal filter of the
    detrended series gives a preliminary seasonal; a trend filter of the seasonally
    adjusted series gives a preliminary trend;
 2. a seasonal filter of the series detrended by the preliminary trend gives the
    seasonal; a trend filter of the seasonally adjusted series gives the trend.

In each pass the seasonal filter is applied to every cycle-subseries, extended to the
points where it is undefined by repeating the nearest estimate of the same phase (which
holds a changing seasonal amplitude fixed over the first and last `p/2` points), and
centred by subtracting its `2 × p` moving average. The noise is `y - trend - seasonal`.
With the default end weights, a linear trend plus a fixed seasonal pattern is reproduced
exactly, including at the ends.

# Keyword arguments
- `period`: seasonal period in units of `t` (required); `p = period / Δt` must be an
  integer, and the series must span at least three cycles.
- `trend`: an [`AbstractTrendSpec`](@ref) for the final trend: the Henderson filter
  [`LocalPolynomialTrend`](@ref) (default) or [`LoessTrend`](@ref). Its window, if
  `nothing`, is derived from `p`, and its default end weights are QL
  `MinimumRevision()` with the ratio estimated from the seasonally adjusted series. Use
  `LocalPolynomialTrend(; window=13, endpoints=musgrave(3.5))` for X-11's Musgrave
  weights.
- `preliminary_trend`: the first-pass trend; `nothing` (default) uses `trend` with the
  same window and QL end weights of ratio `0` (`NoEndpoints()` is not allowed).
- `preliminary_seasonal_filter`, `seasonal_filter`: `(n, m)` for an `S(n × m)` filter,
  or `StableSeasonalFilter`. The defaults are `(3, 3)` and `nothing`, which chooses
  among `S3×3`, `S3×5` and `S3×9` by X-11's global moving seasonality ratio (MSR; S3×3
  below seven cycles). Its thresholds, in units of cycles, were calibrated on monthly
  and quarterly data.
  A filter needing more cycles than available (`n + m - 1`) is replaced by the longest
  one that fits, and by a stable seasonal below five cycles.

Hyperparameters chosen from the data are reported in one `@info` message and stored in
`d.θ.estimated`. Unlike X-13ARIMA-SEATS, there is no extreme-value adjustment, forecast
extension or calendar adjustment.

The rate is the derivative of the trend per unit of `t`.
"""
function fit(
    ::Type{X11Decomp},
    y::AbstractVector{<:Real},
    t::AbstractVector{<:Real}=eachindex(y);
    period::Real,
    trend::AbstractTrendSpec=LocalPolynomialTrend(),
    preliminary_trend::Union{Nothing,AbstractTrendSpec}=nothing,
    preliminary_seasonal_filter::_SeasonalFilterSpec=(3, 3),
    seasonal_filter::_SeasonalFilterSpec=nothing,
)
    y, t, Δt = _uniform_grid(y, t)
    p = _samples_per_period(period, Δt)
    N = length(y)
    cycles = N ÷ p
    cycles ≥ 3 || throw(ArgumentError("need at least 3 seasonal cycles, got $cycles"))
    estimated = (;)
    window = something(trend.window, _default_window(trend, p))
    isnothing(trend.window) && (estimated = (; trend_window=window))
    preliminary_spec = something(preliminary_trend, _preliminary(trend, window))
    preliminary_spec.endpoints isa NoEndpoints &&
        throw(ArgumentError("the preliminary trend needs end weights"))
    preliminary_window = something(preliminary_spec.window, window)

    # Pass 1
    initial_trend = fit(MovingAverage, y; window=p)
    inner, trend₀ = _symmetric_filter(initial_trend.weights, y)
    preliminary_seasonal, preliminary_note = _seasonal_filter(
        preliminary_seasonal_filter, p, length(inner) ÷ p
    )
    if !isnothing(preliminary_note)
        estimated = (;
            estimated...,
            preliminary_seasonal_filter=preliminary_seasonal,
            preliminary_seasonal_filter_note=preliminary_note,
        )
    end
    seasonal = _seasonal(preliminary_seasonal, y[inner] - trend₀, inner, N, p)
    adjusted = y - seasonal
    preliminary_trend_filter, preliminary_trend_values, preliminary_estimated = _trend(
        preliminary_spec, adjusted, preliminary_window
    )
    estimated = (; estimated..., _prefix_keys(:preliminary_, preliminary_estimated)...)

    # Pass 2
    detrended = y - preliminary_trend_values
    if isnothing(seasonal_filter) && cycles < _MSR_MIN_CYCLES
        spec = (3, 3)
        msr_note = "MSR needs $_MSR_MIN_CYCLES cycles, got $cycles; using S3×3"
        estimated = (; estimated..., msr_note)
    elseif isnothing(seasonal_filter)
        spec, msr = _msr_filter(detrended, p)
        estimated = (; estimated..., msr)
    else
        spec = seasonal_filter
    end
    final_seasonal, seasonal_note = _seasonal_filter(spec, p, cycles)
    if isnothing(seasonal_filter) || !isnothing(seasonal_note)
        estimated = (; estimated..., seasonal_filter=final_seasonal)
        isnothing(seasonal_note) ||
            (estimated = (; estimated..., seasonal_filter_note=seasonal_note))
    end
    seasonal = _seasonal(final_seasonal, detrended, 1:N, N, p)
    adjusted = y - seasonal
    trend_filter, trend_values, trend_estimated = _trend(trend, adjusted, window)
    estimated = (; estimated..., trend_estimated...)

    _log_estimated(X11Decomp, estimated)
    filters = (;
        initial_trend,
        preliminary_seasonal,
        preliminary_trend=preliminary_trend_filter,
        seasonal=final_seasonal,
        trend=trend_filter,
    )
    return X11Decomp(
        t,
        Component(trend_values),
        Component(seasonal),
        Component(adjusted - trend_values),
        Component(_trend_rate(trend_values, Δt)),
        (; period=Float64(period), Δt, filters, estimated),
    )
end

function _prefix_keys(prefix::Symbol, nt::NamedTuple)
    return NamedTuple(Symbol(prefix, k) => v for (k, v) in pairs(nt))
end

# Seasonal filter from its specification, replaced by the longest filter that fits when
# there are fewer than n + m - 1 cycles; returns the filter and a note if it was replaced
function _seasonal_filter(::Type{StableSeasonalFilter}, p, _)
    return StableSeasonalFilter(p), nothing
end

function _seasonal_filter((n, m)::Tuple{Integer,Integer}, p, cycles)
    n ≥ 1 && m ≥ 1 && isodd(n) && isodd(m) ||
        throw(ArgumentError("S(n × m) needs odd, positive n and m, got ($n, $m)"))
    n + m - 1 ≤ cycles && return SeasonalMovingAverage(p, n, m), nothing
    for (n′, m′) in ((3, 5), (3, 3))
        if n′ + m′ - 1 ≤ min(cycles, n + m - 2)
            f = SeasonalMovingAverage(p, n′, m′)
            return f, "S$(n)×$(m) needs $(n + m - 1) cycles, got $cycles"
        end
    end
    f = StableSeasonalFilter(p)
    return f, "S$(n)×$(m) needs $(n + m - 1) cycles, got $cycles"
end

# Seasonal component from the detrended values `si` on `inner`: filter each
# cycle-subseries, repeat the nearest estimate of the same phase outside `inner`, and
# centre with a 2 × p moving average (its ends extended as constants)
function _seasonal(filter::AbstractSeasonalFilter, si, inner, N, p)
    s = fill(NaN, N)
    s[inner] = predict(filter, si)
    for t in 1:(first(inner) - 1)
        s[t] = s[t + cld(first(inner) - t, p) * p]
    end
    for t in (last(inner) + 1):N
        s[t] = s[t - cld(t - last(inner), p) * p]
    end
    ma = fit(MovingAverage, s; window=p)
    centre_idx, centre = _symmetric_filter(ma.weights, s)
    offset = [centre[clamp(t - first(centre_idx) + 1, 1, length(centre))] for t in 1:N]
    return s - offset
end

# Global moving seasonality ratio of X-11 (table D9A; Ladiray & Quenneville, 2001): in
# each cycle-subseries, the seasonal S is a 7-term moving average of the detrended values
# (extended at each end by three copies of the mean of the first or last three) and the
# irregular is I = detrended - S; the ratio is Σⱼ Nⱼ Īⱼ / Σⱼ Nⱼ S̄ⱼ, where Īⱼ and S̄ⱼ are the
# mean absolute cycle-to-cycle changes of I and S at phase j and Nⱼ its number of cycles
function _moving_seasonality_ratio(detrended, p)
    ΣI = ΣS = 0.0
    for phase in 1:p
        x = detrended[phase:p:end]
        n = length(x)
        extended = [fill(mean(x[1:3]), 3); x; fill(mean(x[(end - 2):end]), 3)]
        seasonal = [mean(view(extended, k:(k + 6))) for k in 1:n]
        irregular = x - seasonal
        ΣI += n * mean(abs, diff(irregular))
        ΣS += n * mean(abs, diff(seasonal))
    end
    ΣS > 0 && return ΣI / ΣS
    return ΣI > 0 ? Inf : 0.0
end

const _MSR_MIN_CYCLES = 7

# X-11's choice of seasonal filter from the MSR: S3×3 below 2.5, S3×5 in (3.5, 5.5),
# S3×9 above 6.5. The closed grey zones [2.5, 3.5] and [5.5, 6.5] trigger a recomputation
# without the last cycle, up to five times, after which S3×5 is used. Returns the (n, m)
# filter and the last MSR computed.
function _msr_filter(detrended, p)
    msr = NaN
    for attempt in 0:5
        msr = _moving_seasonality_ratio(detrended, p)
        msr < 2.5 && return (3, 3), msr
        3.5 < msr < 5.5 && return (3, 5), msr
        msr > 6.5 && return (3, 9), msr
        attempt < 5 && length(detrended) - p ≥ _MSR_MIN_CYCLES * p || break
        detrended = detrended[1:(end - p)]
    end
    return (3, 5), msr
end
