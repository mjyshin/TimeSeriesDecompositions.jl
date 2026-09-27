"""
    StableSeasonal() <: AbstractSeasonalSmoother

Stable (fixed) seasonal pattern: each point is replaced by the mean of all observations
at the same phase of the cycle.
"""
struct StableSeasonal <: AbstractSeasonalSmoother end

"""
    SeasonalMovingAverage(; n=3, m=5) <: AbstractSeasonalSmoother

The `S(n × m)` seasonal filter of X-11: each cycle-subseries (the observations at one
phase of the cycle) is smoothed by an `m`-term moving average and then an `n`-term moving
average. `n` and `m` must be odd. At the ends of each subseries the averages are
truncated and renormalised, which reproduces constants but differs from X-11's published
asymmetric S3×3 and S3×5 weights.
"""
struct SeasonalMovingAverage <: AbstractSeasonalSmoother
    n::Int
    m::Int
end

function SeasonalMovingAverage(; n::Integer=3, m::Integer=5)
    n ≥ 1 && m ≥ 1 && isodd(n) && isodd(m) ||
        throw(ArgumentError("S(n × m) needs odd, positive n and m, got ($n, $m)"))
    return SeasonalMovingAverage(Int(n), Int(m))
end

Base.show(io::IO, ::StableSeasonal) = print(io, "StableSeasonal()")
function Base.show(io::IO, m::SeasonalMovingAverage)
    return print(io, "SeasonalMovingAverage(n=$(m.n), m=$(m.m))")
end

# Cycles of data needed for one symmetric seasonal estimate
_min_cycles(m::SeasonalMovingAverage) = m.n + m.m - 1
_min_cycles(::StableSeasonal) = 1

# Fitted with the period in units of t on regularly spaced t (period / Δt an integer);
# predict(f, t_new) evaluates it at the observed times
function _fit(
    model::AbstractSeasonalSmoother,
    y::AbstractVector{<:Real},
    t::AbstractVector{<:Real};
    period::Real,
)
    y, t = _sorted_data(y, t)
    Δ = _require_grid(t, nameof(typeof(model)))
    p = _samples_per_period(period, Δ)
    values = _seasonal_values(model, y, p)
    return FittedSmoother(model, t, y, values, (; period=Float64(period), Δ), (;))
end

function predict(
    f::FittedSmoother{<:AbstractSeasonalSmoother}, t_new::AbstractVector{<:Real}
)
    return f.values[_grid_indices(f.t, f.state.Δ, t_new)]
end

# Smooth each cycle-subseries of x, a series with p samples per period
function _seasonal_values(::StableSeasonal, x, p)
    s = similar(x)
    for phase in 1:p
        idx = phase:p:length(x)
        s[idx] .= mean(view(x, idx))
    end
    return s
end

function _seasonal_values(model::SeasonalMovingAverage, x, p)
    inner, outer = fill(1 / model.m, model.m), fill(1 / model.n, model.n)
    s = similar(x)
    for phase in 1:p
        idx = phase:p:length(x)
        s[idx] = _truncated_filter(outer, _truncated_filter(inner, view(x, idx)))
    end
    return s
end

# Centred filter with weights w (odd length); at the ends, the weights falling outside
# the series are dropped and the remainder renormalised to sum to one
function _truncated_filter(w::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    h = (length(w) - 1) ÷ 2
    N = length(y)
    return map(1:N) do t
        js = max(-h, 1 - t):min(h, N - t)
        wt = view(w, js .+ (h + 1))
        return dot(wt, view(y, t .+ js)) / sum(wt)
    end
end
