"""
    StableSeasonal() <: AbstractSeasonalSmoother

Fixed seasonal pattern: the mean of the observations at each phase of the cycle.
"""
struct StableSeasonal <: AbstractSeasonalSmoother end

"""
    SeasonalMovingAverage(; n=3, m=5) <: AbstractSeasonalSmoother

The `S(n × m)` seasonal filter of X-11: an `m`-term then an `n`-term moving average (`n`,
`m` odd) of each cycle-subseries. At the ends of a subseries the averages are truncated and
renormalized, unlike X-11's published asymmetric weights.
"""
struct SeasonalMovingAverage <: AbstractSeasonalSmoother
    n::Int
    m::Int
end

function SeasonalMovingAverage(; n::Integer = 3, m::Integer = 5)
    n ≥ 1 && m ≥ 1 && isodd(n) && isodd(m) ||
        throw(ArgumentError("S(n × m) needs odd, positive n and m, got ($n, $m)"))
    return SeasonalMovingAverage(Int(n), Int(m))
end

Base.show(io::IO, ::StableSeasonal) = print(io, "StableSeasonal()")
function Base.show(io::IO, m::SeasonalMovingAverage)
    return print(io, "SeasonalMovingAverage(n=$(m.n), m=$(m.m))")
end

# Cycles needed for one symmetric estimate
_min_cycles(m::SeasonalMovingAverage) = m.n + m.m - 1
_min_cycles(::StableSeasonal) = 1

function _fit(
        model::AbstractSeasonalSmoother,
        y::AbstractVector{<:Real},
        t::AbstractVector{<:Real};
        period::Real
)
    y, t = _sorted_data(y, t)
    Δ = _require_grid(t, nameof(typeof(model)))
    p = _samples_per_period(period, Δ)
    values = _seasonal_values(model, y, p)
    return FittedSmoother(model, t, y, values, (; period = Float64(period), Δ), (;))
end

"""
    predict(f::FittedSmoother{<:AbstractSeasonalSmoother}, t_new) -> Vector{Float64}

Evaluate the fitted seasonal pattern at times `t_new` within the observed range,
interpolating trigonometrically between phases as `predict(d::X11Decomp, t_new)`.
"""
function predict(
        f::FittedSmoother{<:AbstractSeasonalSmoother}, t_new::AbstractVector{<:Real}
)
    (; period, Δ) = f.state
    _check_in_range(f.t, t_new; atol = 1e-8Δ)
    p = round(Int, period / Δ)
    return map(t_new) do τ
        c = _grid_position(first(f.t), Δ, τ)
        return isinteger(c) ? f.values[Int(c)] : _periodic_interpolation(f.values, c, p)
    end
end

# Trigonometric interpolation at the fractional index c through p consecutive values around
# c (the two nearly centered windows averaged for odd p), clamped into the series
function _periodic_interpolation(v::AbstractVector{Float64}, c::Real, p::Int)
    k = floor(Int, c)
    if iseven(p)
        return _window_interpolation(v, c, p, k - p ÷ 2 + 1)
    end
    return (
        _window_interpolation(v, c, p, k - (p - 1) ÷ 2) +
        _window_interpolation(v, c, p, k - (p - 3) ÷ 2)
    ) / 2
end

function _window_interpolation(v, c, p, first_index)
    lo = clamp(first_index, 1, length(v) - p + 1)
    return sum(v[j] * _periodic_cardinal(c - j, p) for j in lo:(lo + p - 1))
end

# Periodic cardinal (Dirichlet) function: 1 at multiples of p, 0 at other integers
function _periodic_cardinal(x::Real, p::Int)
    s = sinpi(x / p)
    abs(s) < 1e-12 && return 1.0
    return isodd(p) ? sinpi(x) / (p * s) : sinpi(x) / (p * s) * cospi(x / p)
end

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

# Centered filter, truncated and renormalized at the ends
function _truncated_filter(w::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    h = (length(w) - 1) ÷ 2
    N = length(y)
    return map(1:N) do t
        js = max(-h, 1 - t):min(h, N - t)
        wt = view(w, js .+ (h + 1))
        return dot(wt, view(y, t .+ js)) / sum(wt)
    end
end
