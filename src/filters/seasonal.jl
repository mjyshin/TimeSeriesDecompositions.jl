"""
    StableSeasonalFilter <: AbstractSeasonalFilter

Stable (fixed) seasonal pattern: each point is replaced by the mean of all observations
at the same phase of the cycle of `period` samples.

Create with `fit(StableSeasonalFilter, y; period)`.
"""
struct StableSeasonalFilter <: AbstractSeasonalFilter
    period::Int
end

"""
    SeasonalMovingAverage <: AbstractSeasonalFilter

The `S(n × m)` seasonal filter of X-11: each cycle-subseries (the observations at one
phase of the cycle of `period` samples) is smoothed by an `m`-term moving average and
then an `n`-term moving average, truncated and renormalised at its ends.

Create with `fit(SeasonalMovingAverage, y; period, n=3, m=5)`.
"""
struct SeasonalMovingAverage <: AbstractSeasonalFilter
    period::Int
    n::Int
    m::Int
end

"""
    fit(StableSeasonalFilter, y; period) -> StableSeasonalFilter
    fit(SeasonalMovingAverage, y; period, n=3, m=5) -> SeasonalMovingAverage

Seasonal filter for cycles of `period` samples (no data-dependent hyperparameters; `y` is
only validated).
"""
function fit(::Type{StableSeasonalFilter}, y::AbstractVector{<:Real}; period::Integer)
    _check_series(y)
    return StableSeasonalFilter(_check_period(period))
end

function fit(
    ::Type{SeasonalMovingAverage},
    y::AbstractVector{<:Real};
    period::Integer,
    n::Integer=3,
    m::Integer=5,
)
    _check_series(y)
    n ≥ 1 && m ≥ 1 && isodd(n) && isodd(m) ||
        throw(ArgumentError("S(n × m) needs odd, positive n and m, got ($n, $m)"))
    return SeasonalMovingAverage(_check_period(period), Int(n), Int(m))
end

function _check_period(period::Integer)
    period ≥ 2 || throw(DomainError(period, "period must be at least 2 samples"))
    return Int(period)
end

function predict(f::StableSeasonalFilter, y::AbstractVector{<:Real})
    y = _check_series(y)
    x = similar(y)
    for phase in 1:(f.period)
        idx = phase:(f.period):length(y)
        x[idx] .= mean(view(y, idx))
    end
    return x
end

function predict(f::SeasonalMovingAverage, y::AbstractVector{<:Real})
    y = _check_series(y)
    inner, outer = ones(f.m) ./ f.m, ones(f.n) ./ f.n
    x = similar(y)
    for phase in 1:(f.period)
        idx = phase:(f.period):length(y)
        x[idx] = _truncated_filter(outer, _truncated_filter(inner, view(y, idx)))
    end
    return x
end

function Base.show(io::IO, f::StableSeasonalFilter)
    return print(io, "StableSeasonalFilter(period=$(f.period))")
end

function Base.show(io::IO, f::SeasonalMovingAverage)
    return print(io, "SeasonalMovingAverage(period=$(f.period), n=$(f.n), m=$(f.m))")
end
